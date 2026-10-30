#!/usr/bin/env python3
"""Insert a verified solution installer into an actual downloaded service template.

Examples (run from any working directory):
  python dev/build_service_notebook.py --template service-template.ipynb --list-cells
  python dev/build_service_notebook.py --template service-template.ipynb \
      --insert-before 4 --out candidate.ipynb --installer-out installer.py
  python dev/build_service_notebook.py --installer-out installer.py

The last form exports only a reviewable installer, not a notebook. The caller
must supply the real service template for notebook creation; this tool cannot
attest its provenance. Select an insertion point after SOUP_REPO is set and
after any template cell that writes starter solution files, but before the
harness executes. Original cells, outputs, and metadata are retained unchanged.

Only Python source files from this project's solution/ directory are embedded.
The installer writes files but never imports solution, calls apply, or trains.
At runtime, regular starter files at payload paths may be replaced atomically;
unrelated files remain untouched. Symlinks and non-file conflicts are rejected.
Output artifacts are never overwritten with different contents.
"""

import argparse
import base64
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


SOLUTION_DIR = Path(__file__).absolute().parent.parent / "solution"
METADATA_KEY = "soup_solution_installer"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _check_no_symlink(path):
    """Check existing path components without resolving away a symlink."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"Symlinks are not allowed: {part}")
    return path


def _solution_path(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid solution path: {name!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or str(path) != name or len(path.parts) < 2
            or path.parts[0] != "solution" or any(p in (".", "..") for p in path.parts)
            or any(":" in p for p in path.parts) or path.suffix != ".py"):
        raise ValueError(f"Path must name a Python file strictly inside solution/: {name!r}")
    return path


def collect_payload(solution_dir=SOLUTION_DIR):
    """Read actual source bytes, rejecting symlinks throughout the source tree."""
    root = _check_no_symlink(solution_dir)
    if not root.is_dir():
        raise ValueError(f"Solution directory does not exist: {root}")
    files = []

    def visit(directory):
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlinks are not allowed in solution/: {entry}")
            if stat.S_ISDIR(mode):
                visit(entry)
            elif entry.suffix == ".py":
                if not stat.S_ISREG(mode):
                    raise ValueError(f"Solution source is not a regular file: {entry}")
                name = "solution/" + entry.relative_to(root).as_posix()
                _solution_path(name)
                with entry.open("rb") as stream:
                    data = stream.read()
                files.append({"path": name, "size_bytes": len(data), "sha256": _sha256(data),
                              "data_base64": base64.b64encode(data).decode("ascii")})

    visit(root)
    if not files:
        raise ValueError("No Python source files were found in solution/")
    files.sort(key=lambda item: item["path"])
    manifest = [{key: item[key] for key in ("path", "size_bytes", "sha256")} for item in files]
    payload = _canonical({"schema_version": 1, "files": files})
    return payload, manifest


def _install_payload(payload_base64, expected_payload_sha256, expected_manifest_sha256):
    """Self-contained runtime; its source is embedded in the inserted code cell."""
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path, PurePosixPath
    import stat
    import uuid

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    raw_repo = os.environ.get("SOUP_REPO")
    if not raw_repo or not raw_repo.strip():
        raise RuntimeError("SOUP_REPO must be set by the service template before this cell runs")
    repo = Path(raw_repo)
    if not repo.is_absolute() or ".." in repo.parts:
        raise RuntimeError("SOUP_REPO must be an absolute repository path without '..'")

    payload_bytes = base64.b64decode(payload_base64, validate=True)
    if digest(payload_bytes) != expected_payload_sha256:
        raise RuntimeError("Embedded solution payload SHA256 mismatch")
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "files"}:
        raise ValueError("Invalid solution payload schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported solution payload version")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise ValueError("Solution payload must contain source files")

    decoded, manifest, names = [], [], set()
    for item in payload["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256", "data_base64"}:
            raise ValueError("Invalid solution file record")
        name = item["path"]
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("Invalid solution path")
        path = PurePosixPath(name)
        if (path.is_absolute() or str(path) != name or len(path.parts) < 2
                or path.parts[0] != "solution" or any(p in (".", "..") for p in path.parts)
                or any(":" in p for p in path.parts) or path.suffix != ".py"):
            raise ValueError(f"Unsafe solution path: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate solution path: {name}")
        names.add(name)
        data = base64.b64decode(item["data_base64"], validate=True)
        if type(item["size_bytes"]) is not int or item["size_bytes"] != len(data):
            raise ValueError(f"Source size mismatch: {name}")
        if item["sha256"] != digest(data):
            raise ValueError(f"Source SHA256 mismatch: {name}")
        manifest.append({key: item[key] for key in ("path", "size_bytes", "sha256")})
        decoded.append((path.parts[1:], data, name))
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError(f"Conflicting file and directory paths: {name}")
    if digest(canonical(manifest)) != expected_manifest_sha256:
        raise RuntimeError("Embedded solution manifest SHA256 mismatch")

    # Directory descriptors plus O_NOFOLLOW prevent symlink traversal even
    # between checking a path and opening it. The service runtime is Linux.
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("Safe solution installation requires Linux directory descriptors")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW

    def open_repo():
        descriptor = os.open(repo.anchor, directory_flags)
        try:
            for name in repo.parts[1:]:
                next_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def open_parent(solution_descriptor, parts, create):
        descriptor = os.dup(solution_descriptor)
        try:
            for name in parts[:-1]:
                if create:
                    try:
                        os.mkdir(name, mode=0o755, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                try:
                    next_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        os.close(descriptor)
                        return None
                    raise
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def check_leaf(parent, leaf, name):
        try:
            mode = os.stat(leaf, dir_fd=parent, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"Destination conflict (symlink or non-file): {name}")

    def matches(parent, leaf, data):
        try:
            descriptor = os.open(leaf, file_flags, dir_fd=parent)
        except FileNotFoundError:
            return False
        with os.fdopen(descriptor, "rb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode):
                raise RuntimeError(f"Destination is not a regular file: {leaf}")
            if status.st_size != len(data):
                return False
            actual = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
            return actual.hexdigest() == digest(data)

    repo_descriptor = open_repo()
    solution_descriptor = None
    written = unchanged = 0
    try:
        try:
            os.mkdir("solution", mode=0o755, dir_fd=repo_descriptor)
        except FileExistsError:
            pass
        solution_descriptor = os.open("solution", directory_flags, dir_fd=repo_descriptor)
        # Check every existing destination before replacing any file.
        for parts, data, name in decoded:
            parent = open_parent(solution_descriptor, parts, create=False)
            if parent is not None:
                try:
                    check_leaf(parent, parts[-1], name)
                finally:
                    os.close(parent)
        for parts, data, name in decoded:
            parent = open_parent(solution_descriptor, parts, create=True)
            temporary = None
            try:
                check_leaf(parent, parts[-1], name)
                if matches(parent, parts[-1], data):
                    unchanged += 1
                    continue
                temporary = ".soup-install-" + uuid.uuid4().hex + ".tmp"
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o644, dir_fd=parent)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                check_leaf(parent, parts[-1], name)
                os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
                temporary = None
                written += 1
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary, dir_fd=parent)
                    except FileNotFoundError:
                        pass
                os.close(parent)
        # Verify every installed file against the hashed manifest, including
        # files that were already identical and therefore not rewritten.
        for parts, data, name in decoded:
            parent = open_parent(solution_descriptor, parts, create=False)
            if parent is None:
                raise RuntimeError(f"Installed source directory disappeared: {name}")
            try:
                check_leaf(parent, parts[-1], name)
                if not matches(parent, parts[-1], data):
                    raise RuntimeError(f"Installed source SHA256 mismatch: {name}")
            finally:
                os.close(parent)
    finally:
        if solution_descriptor is not None:
            os.close(solution_descriptor)
        os.close(repo_descriptor)
    result = {"payload_sha256": expected_payload_sha256,
              "manifest_sha256": expected_manifest_sha256,
              "files": len(decoded), "written": written, "unchanged": unchanged}
    print(json.dumps({"soup_solution_install": result}, sort_keys=True))
    return result


def make_installer(solution_dir=SOLUTION_DIR, template_sha256=None):
    payload, manifest = collect_payload(solution_dir)
    metadata = {"schema_version": 1, "template_sha256": template_sha256,
                "payload_sha256": _sha256(payload), "manifest_sha256": _sha256(_canonical(manifest)),
                "files": manifest}
    function_name = "_soup_install_solution_" + metadata["payload_sha256"][:16]
    runtime = inspect.getsource(_install_payload).replace("def _install_payload(", "def " + function_name + "(", 1)
    source = ("# Generated solution installer. Writes files only; does not import solution or run training.\n"
              + "# Payload SHA256: " + metadata["payload_sha256"] + "\n"
              + "# Manifest SHA256: " + metadata["manifest_sha256"] + "\n"
              + runtime + "\n" + function_name + "(\n"
              + "    " + repr(base64.b64encode(payload).decode("ascii")) + ",\n"
              + "    " + repr(metadata["payload_sha256"]) + ",\n"
              + "    " + repr(metadata["manifest_sha256"]) + ",\n)\n"
              + "del " + function_name + "\n")
    return source, metadata


def _read_template(path):
    path = _check_no_symlink(path)
    raw = path.read_bytes()

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant in template: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in template: {key}")
            result[key] = value
        return result

    notebook = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValueError("Supply an existing nbformat 4 service training template")
    if not isinstance(notebook.get("cells"), list) or not all(isinstance(c, dict) for c in notebook["cells"]):
        raise ValueError("Template cells must be a list of cell objects")
    if not isinstance(notebook.get("metadata"), dict):
        raise ValueError("Template metadata must be an object")
    if type(notebook.get("nbformat_minor")) is not int:
        raise ValueError("Template nbformat_minor must be an integer")
    return raw, notebook


def _check_artifact(path, data):
    path = _check_no_symlink(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return path, False
    if not stat.S_ISREG(mode) or path.read_bytes() != data:
        raise FileExistsError(f"Refusing to overwrite an existing different artifact: {path}")
    return path, True


def _write_artifacts(artifacts):
    checked, paths = [], set()
    for path, data in artifacts:
        path, identical = _check_artifact(path, data)
        if path in paths:
            raise ValueError(f"Output artifact paths must be distinct: {path}")
        paths.add(path)
        checked.append((path, data, identical))
    for path, data, identical in checked:
        if identical:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _check_no_symlink(path.parent)
        with path.open("xb") as stream:
            stream.write(data)


def build_notebook(template, out, insert_before, *, solution_dir=SOLUTION_DIR, installer_out=None):
    original_raw, original = _read_template(template)
    if type(insert_before) is not int or not 0 <= insert_before <= len(original["cells"]):
        raise ValueError("--insert-before must be between 0 and the original cell count, inclusive")
    if Path(os.path.abspath(template)) == Path(os.path.abspath(out)):
        raise ValueError("The output notebook must be separate from the original template")
    source, metadata = make_installer(solution_dir, _sha256(original_raw))
    cell = {"cell_type": "code", "execution_count": None,
            "metadata": {METADATA_KEY: metadata}, "outputs": [], "source": source.splitlines(keepends=True)}
    if original["nbformat_minor"] >= 5:
        existing_ids = {c.get("id") for c in original["cells"] if isinstance(c.get("id"), str)}
        cell_id = "soup-solution-" + metadata["payload_sha256"][:16]
        while cell_id in existing_ids:
            cell_id += "-x"
        cell["id"] = cell_id
    notebook = copy.deepcopy(original)
    notebook["cells"].insert(insert_before, cell)
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    roundtrip = json.loads(serialized)
    del roundtrip["cells"][insert_before]
    if roundtrip != original:
        raise RuntimeError("Template preservation check failed; no output will be written")
    artifacts = [(out, serialized)]
    if installer_out is not None:
        artifacts.append((installer_out, source.encode("utf-8")))
    _write_artifacts(artifacts)
    return {**metadata, "out": str(Path(out).absolute()), "insert_before": insert_before,
            "notebook_sha256": _sha256(serialized), "installer_sha256": _sha256(source.encode("utf-8")),
            "original_cell_count": len(original["cells"]), "template_roundtrip_verified": True}


def list_cells(template):
    raw, notebook = _read_template(template)
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if not isinstance(source, str):
            raise ValueError(f"Cell {index} has a non-text source")
        preview = next((line.strip() for line in source.splitlines() if line.strip()), "")
        cells.append({"index": index, "cell_type": cell.get("cell_type"),
                      "id": cell.get("id"), "source_preview": preview[:180]})
    return {"template_sha256": _sha256(raw), "cell_count": len(cells),
            "append_index": len(cells), "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, help="Actual downloaded service template; never generated by this tool")
    parser.add_argument("--out", type=Path, help="New notebook path; different existing contents are not overwritten")
    parser.add_argument("--insert-before", type=int, help="Zero-based original cell index; cell count means append")
    parser.add_argument("--list-cells", action="store_true", help="List original indices/previews without writing anything")
    parser.add_argument("--installer-out", type=Path, help="Optional standalone installer source, also available without a template")
    args = parser.parse_args(argv)
    if args.list_cells:
        if args.template is None or args.out is not None or args.insert_before is not None or args.installer_out is not None:
            parser.error("--list-cells requires only --template and does not write outputs")
    elif args.out is not None or args.insert_before is not None:
        if args.template is None or args.out is None or args.insert_before is None:
            parser.error("Building a notebook requires --template, --out, and --insert-before")
    elif args.installer_out is None:
        parser.error("Choose --list-cells, notebook creation, or --installer-out")
    try:
        if args.list_cells:
            result = list_cells(args.template)
        elif args.out is not None:
            result = build_notebook(args.template, args.out, args.insert_before, installer_out=args.installer_out)
        else:
            template_hash = _sha256(_read_template(args.template)[0]) if args.template is not None else None
            source, metadata = make_installer(template_sha256=template_hash)
            data = source.encode("utf-8")
            _write_artifacts([(args.installer_out, data)])
            result = {**metadata, "installer_out": str(args.installer_out.absolute()),
                      "installer_sha256": _sha256(data)}
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
