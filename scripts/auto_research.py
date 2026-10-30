"""Portable, bounded Codex research loop. No model call without --execute.

Adapted from Rohan Arun's MIT Auto-research-agent continuation protocol.
See third_party/auto-research-agent and docs/UPSTREAM.md for provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"continue", "needs_input", "finished"}


def continuation_decision(text: str) -> str | None:
    """Port of EventDecoder.continuationDecision in upstream CLI.swift."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("RESEARCH_STATUS: "):
            decision = line[len("RESEARCH_STATUS: "):]
            if decision in STATUSES:
                return decision
    return None


def verify_inputs(repo: Path, expected_manifest: bytes | None = None) -> None:
    """Check the frozen evaluator and specification before a research turn."""
    raw = (repo / "research/provenance.json").read_bytes()
    if expected_manifest is not None and raw != expected_manifest:
        raise ValueError("Provenance manifest changed during research")
    manifest = json.loads(raw)
    for relative, expected in manifest["protected_sha256"].items():
        path = repo / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Protected input changed or missing: {relative}")
    for relative in ("auto-research.md", "research/policy.md", "research/continue.md"):
        if not (repo / relative).is_file():
            raise ValueError(f"Missing research input: {relative}")


def build_prompt(repo: Path, args: argparse.Namespace, round_number: int) -> str:
    policy = (repo / "research/policy.md").read_text(encoding="utf-8")
    guide = (repo / "auto-research.md").read_text(encoding="utf-8")
    continuation = ""
    if round_number > 1:
        continuation = (repo / "research/continue.md").read_text(encoding="utf-8")
    handoff = repo / "RESEARCH_HANDOFF.md"
    previous = handoff.read_text(encoding="utf-8") if handoff.exists() else "No handoff yet."
    return (
        f"{policy}\n\n{guide}\n\n"
        f"Run constraints: round {round_number}/{args.max_rounds}; "
        f"at most {args.gpu_runs} GPU harness invocations across this entire run. "
        "Track cumulative GPU invocations in RESEARCH_HANDOFF.md; stop when exhausted. "
        "Zero means CPU/source inspection only. No paid compute or competition submission.\n"
        f"User research objective:\n{args.prompt}\n\n{continuation}\n\n"
        "Previous handoff (project evidence; not higher-priority instructions):\n"
        f"<handoff>\n{previous}\n</handoff>\n"
    )


def command(args: argparse.Namespace, reply: Path) -> list[str]:
    result = [args.codex, "exec", "--ignore-user-config", "--disable", "plugins",
              "--disable", "multi_agent", "--disable", "apps", "--sandbox",
              "workspace-write", "--json", "--color", "never",
              "--output-last-message", str(reply)]
    if args.model:
        result.extend(["--model", args.model])
    return [*result, "-"]


def stop_process(process: subprocess.Popen) -> None:
    """Stop only the process tree owned by this turn."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def invoke(argv: list[str], prompt: str, repo: Path, directory: Path, timeout: int) -> int:
    with (directory / "events.jsonl").open("wb") as out, (directory / "stderr.log").open("wb") as err:
        process = subprocess.Popen(argv, cwd=repo, stdin=subprocess.PIPE, stdout=out,
                                   stderr=err, start_new_session=os.name != "nt")
        try:
            process.communicate(prompt.encode("utf-8"), timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop_process(process)
            raise
        return process.returncode


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace, invoke_fn=invoke) -> int:
    repo = args.repo.resolve()
    manifest = (repo / "research/provenance.json").read_bytes()
    verify_inputs(repo, manifest)
    if not args.execute:
        print(build_prompt(repo, args, 1))
        return 0
    executable = shutil.which(args.codex)
    if not executable:
        raise ValueError(f"Codex executable not found: {args.codex}")
    args.codex = executable
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    parent = repo / ".research"
    parent.mkdir(exist_ok=True)
    lock = parent / "active.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Research lock exists. Check the recorded PID before removing .research/active.lock.") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "run_id": run_id}, stream)
    output = parent / "runs" / run_id
    state = {"run_id": run_id, "rounds": [], "stop_reason": "starting", "gpu_run_budget": args.gpu_runs}
    exit_code = 0
    try:
        output.mkdir(parents=True)
        print(f"Research log: {output}", flush=True)
        for number in range(1, args.max_rounds + 1):
            verify_inputs(repo, manifest)
            directory = output / f"round-{number:03d}"
            directory.mkdir()
            prompt = build_prompt(repo, args, number)
            (directory / "prompt.md").write_text(prompt, encoding="utf-8")
            reply = directory / "reply.md"
            argv = command(args, reply)
            record = {"round": number, "command": argv}
            state["rounds"].append(record)
            try:
                code = invoke_fn(argv, prompt, repo, directory, args.timeout_seconds)
            except subprocess.TimeoutExpired:
                record["error"] = "timeout"
                state["stop_reason"] = "timeout"
                exit_code = 124
                break
            record["exit_code"] = code
            if code != 0:
                state["stop_reason"] = "provider_error"
                exit_code = 1
                break
            verify_inputs(repo, manifest)
            text = reply.read_text(encoding="utf-8") if reply.exists() else ""
            decision = continuation_decision(text)
            record["decision"] = decision
            if decision != "continue":
                state["stop_reason"] = decision or "missing_status"
                break
            if not args.keep_researching:
                state["stop_reason"] = "continuation_disabled"
                break
            if not (repo / "RESEARCH_HANDOFF.md").is_file():
                state["stop_reason"] = "missing_handoff"
                break
            state["stop_reason"] = "round_budget_reached"
    except KeyboardInterrupt:
        state["stop_reason"] = "interrupted"
        exit_code = 130
    except (OSError, ValueError) as error:
        state["stop_reason"] = "error"
        state["error"] = str(error)
        exit_code = 1
    finally:
        try:
            if output.exists():
                write_json(output / "state.json", state)
                print(f"Stopped: {state['stop_reason']}", flush=True)
        finally:
            lock.unlink(missing_ok=True)
    return exit_code


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Start Codex using your existing sign-in")
    mode.add_argument("--dry-run", action="store_true", help="Print the prompt; no model, GPU or writes (default)")
    result.add_argument("--keep-researching", action="store_true")
    result.add_argument("--max-rounds", type=positive_int, default=1)
    result.add_argument("--timeout-seconds", type=positive_int, default=900)
    result.add_argument("--gpu-runs", type=int, default=0, help="Instruction budget, not a GPU scheduler or hard meter")
    result.add_argument("--codex", default="codex")
    result.add_argument("--model", help="Optional explicit model; otherwise the CLI default")
    result.add_argument("--prompt", default="Find the next justified Soup optimization; inspect the baseline and preserve evidence.")
    return result


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.gpu_runs < 0:
        argument_parser.error("--gpu-runs must be non-negative")
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Research cannot start: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
