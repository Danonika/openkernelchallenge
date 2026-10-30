"""Focused stdlib tests; fixture notebooks are synthetic and never submissions.

Run: python -m unittest discover -s dev -p test_notebook_builder.py -v
"""

import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("build_service_notebook", Path(__file__).with_name("build_service_notebook.py"))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class NotebookBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.solution = self.root / "sources" / "solution"
        (self.solution / "ops").mkdir(parents=True)
        self.contents = {"__init__.py": b"# original bytes\r\ndef apply():\r\n    pass\r\n",
                         "ops/kernel.py": "# Unicode source: λ\nraise RuntimeError('must not be imported')\n".encode()}
        for name, data in self.contents.items():
            (self.solution / name).write_bytes(data)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.template = self.root / "fixture-template.ipynb"
        self.original = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3"}, "unfamiliar": {"nested": [1, False, None]}},
            "cells": [
                {"id": "intro", "cell_type": "markdown", "metadata": {"custom": "preserve"},
                 "source": ["# Test fixture only\n"], "attachments": {"x.txt": {"text/plain": "abc"}}},
                {"id": "harness", "cell_type": "code", "metadata": {"tags": ["harness", "keep"]},
                 "execution_count": 7, "source": "# harness placeholder\nprint('fixture')\n",
                 "outputs": [{"output_type": "display_data", "metadata": {"custom": 3},
                              "data": {"text/plain": ["saved output\n"], "application/json": {"x": [1, 2]}}}]},
                {"id": "reports", "cell_type": "code", "metadata": {}, "execution_count": None,
                 "source": ["# report placeholder\n"], "outputs": []},
            ],
        }
        self.original_bytes = json.dumps(self.original, indent=3).encode()
        self.template.write_bytes(self.original_bytes)

    def run_installer(self, source):
        output = io.StringIO()
        with patch.dict(os.environ, {"SOUP_REPO": str(self.repo)}), contextlib.redirect_stdout(output):
            exec(compile(source, "<test-installer>", "exec"), {})
        return json.loads(output.getvalue())["soup_solution_install"]

    def raw_record(self, name, data=b"pass\n"):
        return {"path": name, "size_bytes": len(data), "sha256": builder._sha256(data),
                "data_base64": base64.b64encode(data).decode()}

    def run_records(self, records, *, manifest_hash=None, payload_hash=None):
        manifest = [{k: item[k] for k in ("path", "size_bytes", "sha256")} for item in records]
        payload = builder._canonical({"schema_version": 1, "files": records})
        with patch.dict(os.environ, {"SOUP_REPO": str(self.repo)}), contextlib.redirect_stdout(io.StringIO()):
            return builder._install_payload(base64.b64encode(payload).decode(),
                                            payload_hash or builder._sha256(payload),
                                            manifest_hash or builder._sha256(builder._canonical(manifest)))

    def test_roundtrip_preserves_every_original_cell_and_metadata(self):
        out, installer = self.root / "candidate.ipynb", self.root / "installer.py"
        result = builder.build_notebook(self.template, out, 1, solution_dir=self.solution, installer_out=installer)
        notebook = json.loads(out.read_bytes())
        added = notebook["cells"].pop(1)
        self.assertEqual(notebook, self.original)
        self.assertEqual(self.template.read_bytes(), self.original_bytes)
        self.assertEqual(added["cell_type"], "code")
        self.assertEqual(added["outputs"], [])
        self.assertIsNone(added["execution_count"])
        self.assertEqual("".join(added["source"]).encode(), installer.read_bytes())
        self.assertEqual(result["template_sha256"], builder._sha256(self.original_bytes))
        self.assertEqual(result["notebook_sha256"], builder._sha256(out.read_bytes()))
        self.assertTrue(result["template_roundtrip_verified"])
        self.assertEqual(added["metadata"][builder.METADATA_KEY]["files"], result["files"])
        listing = builder.list_cells(self.template)
        self.assertEqual([c["index"] for c in listing["cells"]], [0, 1, 2])
        self.assertEqual(listing["cells"][1]["source_preview"], "# harness placeholder")

    def test_installer_replaces_starter_preserves_unrelated_and_is_idempotent(self):
        destination = self.repo / "solution"
        destination.mkdir()
        (destination / "__init__.py").write_text("# old official starter\n")
        (destination / "unrelated.txt").write_bytes(b"keep me")
        source, metadata = builder.make_installer(self.solution)
        first = self.run_installer(source)
        self.assertEqual(first["written"], len(self.contents))
        self.assertEqual(first["payload_sha256"], metadata["payload_sha256"])
        for name, data in self.contents.items():
            self.assertEqual((destination / name).read_bytes(), data)
        self.assertEqual((destination / "unrelated.txt").read_bytes(), b"keep me")
        tracked = destination / "__init__.py"
        os.utime(tracked, ns=(1234567890000000000, 1234567890000000000))
        second = self.run_installer(source)
        self.assertEqual(second["written"], 0)
        self.assertEqual(second["unchanged"], len(self.contents))
        self.assertEqual(tracked.stat().st_mtime_ns, 1234567890000000000)
        self.assertFalse(list(destination.rglob(".soup-install-*.tmp")))

    def test_rejects_traversal_duplicates_and_file_directory_conflicts(self):
        unsafe = ["../escape.py", "/solution/escape.py", "solution/../escape.py",
                  "solution/sub/../../escape.py", "solution//escape.py", "solution/./escape.py",
                  "solution\\escape.py", "elsewhere/escape.py", "solution/C:/escape.py"]
        for name in unsafe:
            with self.subTest(path=name), self.assertRaises(ValueError):
                self.run_records([self.raw_record(name)])
        for names in (("solution/a.py", "solution/a.py"), ("solution/a.py", "solution/a.py/b.py")):
            with self.subTest(paths=names), self.assertRaises(ValueError):
                self.run_records([self.raw_record(name) for name in names])
        self.assertEqual(list(self.repo.iterdir()), [])
        self.assertFalse((self.root / "escape.py").exists())

    def test_rejects_destination_symlink_or_directory_before_any_replacement(self):
        destination = self.repo / "solution"
        destination.mkdir()
        first = destination / "__init__.py"
        first.write_bytes(b"old starter")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "kernel.py").write_bytes(b"external data")
        source, _ = builder.make_installer(self.solution)
        (destination / "ops").symlink_to(outside, target_is_directory=True)
        with self.assertRaises((OSError, RuntimeError)):
            self.run_installer(source)
        self.assertEqual(first.read_bytes(), b"old starter")
        self.assertEqual((outside / "kernel.py").read_bytes(), b"external data")
        (destination / "ops").unlink()
        (destination / "ops").mkdir()
        leaf = destination / "ops" / "kernel.py"
        leaf.mkdir()
        with self.assertRaises(RuntimeError):
            self.run_installer(source)
        self.assertEqual(first.read_bytes(), b"old starter")
        leaf.rmdir()
        leaf.symlink_to(outside / "kernel.py")
        with self.assertRaises(RuntimeError):
            self.run_installer(source)
        self.assertEqual(first.read_bytes(), b"old starter")
        self.assertEqual((outside / "kernel.py").read_bytes(), b"external data")

    def test_rejects_symlinked_solution_root_and_source(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.repo / "solution").symlink_to(outside, target_is_directory=True)
        source, _ = builder.make_installer(self.solution)
        with self.assertRaises(OSError):
            self.run_installer(source)
        self.assertEqual(list(outside.iterdir()), [])
        external_file = self.root / "external.py"
        external_file.write_bytes(b"pass\n")
        (self.solution / "linked.py").symlink_to(external_file)
        with self.assertRaises(ValueError):
            builder.collect_payload(self.solution)

    def test_checks_payload_manifest_and_individual_source_hashes(self):
        records = [self.raw_record("solution/a.py")]
        with self.assertRaises(RuntimeError):
            self.run_records(records, payload_hash="0" * 64)
        with self.assertRaises(RuntimeError):
            self.run_records(records, manifest_hash="0" * 64)
        records[0]["data_base64"] = base64.b64encode(b"fail\n").decode()
        with self.assertRaises(ValueError):
            self.run_records(records)
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_missing_environment_fails_without_implicit_repository(self):
        source, _ = builder.make_installer(self.solution)
        environment = dict(os.environ)
        environment.pop("SOUP_REPO", None)
        with patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(RuntimeError, "SOUP_REPO"):
            exec(compile(source, "<test-installer>", "exec"), {})
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_output_conflicts_are_non_destructive_and_identical_rebuild_is_allowed(self):
        out = self.root / "candidate.ipynb"
        out.write_bytes(b"existing artifact")
        with self.assertRaises(FileExistsError):
            builder.build_notebook(self.template, out, 1, solution_dir=self.solution)
        self.assertEqual(out.read_bytes(), b"existing artifact")
        out.unlink()
        blocker = self.root / "installer.py"
        blocker.write_bytes(b"keep existing installer")
        with self.assertRaises(FileExistsError):
            builder.build_notebook(self.template, out, 1, solution_dir=self.solution, installer_out=blocker)
        self.assertFalse(out.exists())
        self.assertEqual(blocker.read_bytes(), b"keep existing installer")
        first = builder.build_notebook(self.template, out, 3, solution_dir=self.solution)
        second = builder.build_notebook(self.template, out, 3, solution_dir=self.solution)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            builder.build_notebook(self.template, self.template, 0, solution_dir=self.solution)
        self.assertEqual(self.template.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
