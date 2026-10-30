"""CPU installation checks only; these are not Soup's official regressions.

Run from the candidate root:
    python -m unittest discover -s dev -p test_metrics_install.py -v

The tests compile only the installer and wrapper function definitions from
solution/metrics.py.  Controlled stand-ins supply their dependency interfaces.
No real torch, Triton, TRL, CUDA, or task modules are imported or exercised.
These tests establish installation/fallback behavior, not kernel correctness,
numerical equivalence, performance, or challenge eligibility.
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "solution" / "metrics.py"


class MetricsInstallCPUChecks(unittest.TestCase):
    def setUp(self):
        def entropy_from_logits(logits, chunk_size=128):
            return ("native", logits, chunk_size)

        entropy_from_logits.__module__ = "trl.trainer.utils"

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return None

        compute_loss.__module__ = "trl.trainer.sft_trainer"
        self.original = entropy_from_logits
        self.trainer = type("SFTTrainer", (), {"compute_loss": compute_loss})
        self.utils = types.ModuleType("trl.trainer.utils")
        self.sft = types.ModuleType("trl.trainer.sft_trainer")
        self.utils.entropy_from_logits = self.original
        self.sft.entropy_from_logits = self.original
        self.sft.SFTTrainer = self.trainer
        self.import_error = False

        def controlled_import(name):
            if self.import_error:
                raise ImportError("controlled optional dependency is unavailable")
            if name == "trl.trainer.utils":
                return self.utils
            if name == "trl.trainer.sft_trainer":
                return self.sft
            raise AssertionError(f"Unexpected dependency import: {name}")

        # Deliberately avoid importing solution.metrics and its GPU packages.
        # Execute its exact current installer/wrapper definitions, with only
        # these controlled names available for dependency and kernel access.
        source_tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        function_names = {"install", "_make_entropy_wrapper"}
        definitions = [
            node for node in source_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        ]
        self.assertEqual({node.name for node in definitions}, function_names)
        self.namespace = {
            "functools": functools,
            "importlib": types.SimpleNamespace(import_module=controlled_import),
            "inspect": inspect,
            "types": types,
            "_can_use_entropy": lambda logits, chunk_size: False,
            "_accuracy_without_copy": self._unexpected_accuracy_rewrite,
        }
        extracted = ast.Module(body=definitions, type_ignores=[])
        exec(compile(extracted, str(SOURCE), "exec"), self.namespace)

    @staticmethod
    def _unexpected_accuracy_rewrite(original):
        raise AssertionError("Accuracy rewriting must stay disabled by default")

    def assert_unchanged_fallback(self):
        # Snapshot after constructing the unsupported interface.  The installer
        # must leave even that incomplete interface exactly as it found it.
        utils_before = dict(self.utils.__dict__)
        sft_before = dict(self.sft.__dict__)
        trainer_before = dict(self.trainer.__dict__)
        result = self.namespace["install"]()
        self.assertEqual(result, {"entropy": False, "accuracy_no_copy": False})
        self.assertEqual(dict(self.utils.__dict__), utils_before)
        self.assertEqual(dict(self.sft.__dict__), sft_before)
        self.assertEqual(dict(self.trainer.__dict__), trainer_before)

    def test_fallback_missing_dependency_import(self):
        self.import_error = True
        self.assert_unchanged_fallback()

    def test_fallback_missing_utils_helper(self):
        del self.utils.entropy_from_logits
        self.assert_unchanged_fallback()

    def test_fallback_missing_sft_binding(self):
        del self.sft.entropy_from_logits
        self.assert_unchanged_fallback()

    def test_fallback_missing_trainer_class(self):
        del self.sft.SFTTrainer
        self.assert_unchanged_fallback()

    def test_fallback_mock_module(self):
        self.utils = Mock()
        self.assert_unchanged_fallback()

    def test_fallback_mock_entropy_helper(self):
        self.utils.entropy_from_logits = Mock()
        self.assert_unchanged_fallback()

    def test_fallback_mock_trainer_class(self):
        self.sft.SFTTrainer = Mock()
        self.assert_unchanged_fallback()

    def test_fallback_missing_trainer_method(self):
        self.sft.SFTTrainer = type("TrainerWithoutComputeLoss", (), {})
        self.assert_unchanged_fallback()

    def test_fallback_mismatched_entropy_bindings(self):
        self.sft.entropy_from_logits = lambda logits, chunk_size=128: None
        self.assert_unchanged_fallback()

    def test_installation_is_idempotent_and_accuracy_stays_disabled(self):
        expected = {"entropy": True, "accuracy_no_copy": False}
        original_method = self.trainer.compute_loss
        self.assertEqual(self.namespace["install"](), expected)
        installed = self.utils.entropy_from_logits
        self.assertIs(self.sft.entropy_from_logits, installed)
        self.assertIs(installed._soup_entropy_original, self.original)
        self.assertEqual(installed("input", chunk_size=17), ("native", "input", 17))
        self.assertEqual(self.namespace["install"](), expected)
        self.assertIs(self.utils.entropy_from_logits, installed)
        self.assertIs(self.sft.entropy_from_logits, installed)
        self.assertIs(self.trainer.compute_loss, original_method)

    def test_simulated_kernel_runtime_error_propagates(self):
        class SimulatedKernelError(RuntimeError):
            pass

        def failing_kernel(logits):
            raise SimulatedKernelError("controlled simulated kernel failure")

        self.namespace["_can_use_entropy"] = lambda logits, chunk_size: True
        self.namespace["_entropy_forward"] = failing_kernel
        self.namespace["install"]()
        with self.assertRaisesRegex(SimulatedKernelError, "controlled simulated kernel failure"):
            self.utils.entropy_from_logits("input")


if __name__ == "__main__":
    unittest.main()
