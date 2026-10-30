"""Dependency-free package checks; this is not the official rule checker."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER_SHA256 = "e665910faeca6091c958ed7a191604484a44216f51d6fa069f5f48220e7472e6"
# From pinned harness/common.py. This development script never edits/imports
# the harness or substitutes for its runtime checks.
BANNED = frozenset((
    "unsloth", "unsloth_zoo", "liger_kernel", "xformers", "flash_attn",
    "flashinfer", "apex", "torchao", "bitblas", "gemlite", "hqq",
    "cut_cross_entropy", "deepspeed", "vllm", "awq", "auto_gptq",
    "gptqmodel", "exllamav2", "marlin", "quack", "tilelang",
))


def check(root=ROOT):
    root = Path(root)
    solution = root / "solution"
    errors = []
    files = {}
    original_kernels = []
    example_kernels = []
    trees = {}
    for path in sorted(solution.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"Symlink is not a deliverable source file: {relative}")
            continue
        source = path.read_bytes()
        files[relative] = hashlib.sha256(source).hexdigest()
        try:
            tree = ast.parse(source, filename=relative)
            compile(tree, relative, "exec")
        except SyntaxError as exc:
            errors.append(f"Invalid Python in {relative}: {exc}")
            continue
        trees[relative] = tree
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports = [node.module.split('.')[0]]
            if BANNED.intersection(imports):
                errors.append(f"Forbidden import in {relative}: {imports}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
                if "triton.jit" in decorators:
                    target = example_kernels if node.name.startswith("example_") else original_kernels
                    target.append(f"{relative}:{node.name}")
        text = source.decode("utf-8")
        for pattern, reason in (
            (r"\b(?:CUDAGraph|make_graphed_callables)\b", "CUDA graph API"),
            (r"PYTORCH_(?:CUDA_)?ALLOC_CONF", "allocator setting"),
            (r"(?:Authorization|authorization)\s*[:=]\s*[\"'](?:Bearer|Basic)\s+\S+", "embedded authorization"),
        ):
            if re.search(pattern, text):
                errors.append(f"Unexpected {reason} in {relative}")

    if files.get("solution/swiglu.py") != STARTER_SHA256:
        errors.append("Supplied example swiglu.py is not byte-identical to the credited starter")
    initializer = trees.get("solution/__init__.py")
    entries = [] if initializer is None else [
        node for node in initializer.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    ]
    if len(entries) != 1:
        errors.append("Expected one solution.apply function")
    else:
        args = entries[0].args
        if args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg:
            errors.append("solution.apply must take zero arguments")
        body = ast.unparse(entries[0])
        if "install_patches()" not in body:
            errors.append("solution.apply does not call install_patches")

    # Test that importing the entry point itself is inert, without requiring
    # or faking torch. apply() is deliberately not executed in this check.
    if not errors:
        spec = importlib.util.spec_from_file_location(
            "_soup_candidate_static_entry", solution / "__init__.py",
            submodule_search_locations=[str(solution)],
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "apply", None)):
            errors.append("Imported solution has no callable apply")

    return {
        "status": "passed" if not errors else "failed",
        "scope": "static source and entry-point checks only; no CUDA or official regression execution",
        "files_sha256": files,
        "original_jit_definitions": original_kernels,
        "credited_example_definitions": example_kernels,
        "original_kernel_runtime_share": "unmeasured",
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "passed" else 1)
