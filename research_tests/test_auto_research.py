"""Offline behavioral checks; these never invoke a model or CUDA."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("auto_research", ROOT / "scripts/auto_research.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo with spaces"
        self.repo.mkdir()
        manifest = json.loads((ROOT / "research/provenance.json").read_text(encoding="utf-8"))
        files = list(manifest["protected_sha256"]) + [
            "research/provenance.json", "auto-research.md", "research/policy.md", "research/continue.md"]
        for name in files:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.args = ar.parser().parse_args(["--repo", str(self.repo), "--codex", sys.executable,
                                           "--execute", "--keep-researching", "--max-rounds", "3"])
        self.calls = []

    def execute(self, replies, mutation=None):
        def fake(argv, prompt, repo, directory, timeout):
            index = len(self.calls)
            self.calls.append(prompt)
            result = replies[index]
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, int):
                return result
            (directory / "reply.md").write_text(result, encoding="utf-8")
            (repo / "RESEARCH_HANDOFF.md").write_text(f"Evidence round {index + 1}", encoding="utf-8")
            if mutation:
                mutation(repo)
            return 0
        with contextlib.redirect_stdout(io.StringIO()):
            code = ar.run(self.args, invoke_fn=fake)
        states = list((self.repo / ".research/runs").glob("*/state.json"))
        state = json.loads(states[-1].read_text(encoding="utf-8"))
        self.assertFalse((self.repo / ".research/active.lock").exists())
        return code, state

    def test_continue_passes_handoff_to_next_round_then_finishes(self):
        code, state = self.execute(["Evidence\nRESEARCH_STATUS: continue", "RESEARCH_STATUS: finished"])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("Evidence round 1", self.calls[1])
        self.assertEqual(state["stop_reason"], "finished")

    def test_round_limit_is_hard(self):
        code, state = self.execute(["RESEARCH_STATUS: continue"] * 3)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(state["stop_reason"], "round_budget_reached")

    def test_continuation_is_opt_in(self):
        self.args.keep_researching = False
        _, state = self.execute(["RESEARCH_STATUS: continue"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(state["stop_reason"], "continuation_disabled")

    def test_natural_language_does_not_trigger_continuation(self):
        _, state = self.execute(["I can continue once GPU access is available."])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(state["stop_reason"], "missing_status")

    def test_blocker_stops(self):
        _, state = self.execute(["RESEARCH_STATUS: needs_input"])
        self.assertEqual(state["stop_reason"], "needs_input")

    def test_provider_error_stops_and_retains_exit_code(self):
        code, state = self.execute([7])
        self.assertEqual(code, 1)
        self.assertEqual(state["stop_reason"], "provider_error")
        self.assertEqual(state["rounds"][0]["exit_code"], 7)

    def test_timeout_retains_record_and_releases_lock(self):
        code, state = self.execute([subprocess.TimeoutExpired("fake", 1)])
        self.assertEqual(code, 124)
        self.assertEqual(state["stop_reason"], "timeout")

    def test_protected_evaluator_change_stops(self):
        code, state = self.execute(["RESEARCH_STATUS: continue"],
                                  lambda repo: (repo / "task/task.yaml").write_text("changed"))
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Protected input changed", state["error"])

    def test_missing_handoff_stops(self):
        _, state = self.execute(["RESEARCH_STATUS: continue"],
                               lambda repo: (repo / "RESEARCH_HANDOFF.md").unlink())
        self.assertEqual(state["stop_reason"], "missing_handoff")

    def test_manifest_cannot_be_replaced_mid_run(self):
        code, state = self.execute(["RESEARCH_STATUS: continue"],
            lambda repo: (repo / "research/provenance.json").write_text('{"protected_sha256": {}}'))
        self.assertEqual(code, 1)
        self.assertIn("manifest changed", state["error"])

    def test_preview_never_calls_provider_or_creates_run(self):
        self.args.execute = False
        def forbidden(*args):
            self.fail("Preview invoked the provider")
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            self.assertEqual(ar.run(self.args, forbidden), 0)
        self.assertIn("at most 0 GPU", buffer.getvalue())
        self.assertFalse((self.repo / ".research").exists())

    def test_existing_lock_prevents_a_second_run(self):
        (self.repo / ".research").mkdir()
        lock = self.repo / ".research/active.lock"
        lock.write_text("existing owner")
        with self.assertRaisesRegex(ValueError, "Research lock exists"):
            ar.run(self.args, lambda *args: self.fail("Concurrent provider"))
        self.assertEqual(lock.read_text(), "existing owner")

    def test_real_process_receives_utf8_stdin_and_writes_logs(self):
        directory = self.repo / "process"
        directory.mkdir()
        script = "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data); sys.stderr.write('diagnostic')"
        code = ar.invoke([sys.executable, "-c", script], "ТЗ Soup", self.repo, directory, 10)
        self.assertEqual(code, 0)
        self.assertEqual((directory / "events.jsonl").read_text(encoding="utf-8"), "ТЗ Soup")
        self.assertEqual((directory / "stderr.log").read_text(), "diagnostic")

    def test_real_process_timeout_terminates(self):
        directory = self.repo / "timeout"
        directory.mkdir()
        with self.assertRaises(subprocess.TimeoutExpired):
            ar.invoke([sys.executable, "-c", "import time; time.sleep(60)"], "", self.repo, directory, 1)

    def test_prompt_is_not_a_shell_argument(self):
        self.args.prompt = 'Soup & echo sensitive $(whoami)'
        argv = ar.command(self.args, self.repo / "reply.md")
        self.assertNotIn(self.args.prompt, argv)
        self.assertEqual(argv[-1], "-")


if __name__ == "__main__":
    unittest.main()
