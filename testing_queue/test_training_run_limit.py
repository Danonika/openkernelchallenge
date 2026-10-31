"""Deadline boundary and process cleanup tests without spending GPU time."""
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock
from training_run_limit import TrainingRunLimitExceeded, wrap_run_logged
from notebook_worker import Worker, training_process_deadline


class Poller:
    peak, start = 100, 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class TrainingLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'base_1' / 'train.log'
        self.path.parent.mkdir()
        self.marker = self.root / 'training_limit.json'
        self.process = Mock()
        self.process.wait.return_value = 0
        self.original = Mock(return_value=(0, 100, 0, 0))
        self.common = types.SimpleNamespace(run_logged=self.original, GpuPoller=Poller,
            subprocess=types.SimpleNamespace(Popen=Mock(return_value=self.process),
                STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
            REPO=self.root, ENV={}, kill_group=Mock())

    def tearDown(self):
        self.temp.cleanup()

    def run_training(self, elapsed, timeout=900, error=False):
        values = iter([100.0, 100.2, 100.0 + elapsed])
        wrapped = wrap_run_logged(self.common, clock=lambda: next(values), failure_path=self.marker)
        if error:
            self.process.wait.side_effect = subprocess.TimeoutExpired('train', 80)
        return wrapped(['python', '/repo/harness/_train.py'], self.path, timeout)

    def test_79_16_seconds_passes(self):
        self.assertEqual(self.run_training(79.16), (0, 79.16, 100, 0))
        self.assertAlmostEqual(self.process.wait.call_args.kwargs['timeout'], 79.8)
        self.assertFalse(self.marker.exists())
        self.common.kill_group.assert_called_once_with(self.process)

    def test_exactly_80_seconds_passes(self):
        self.assertEqual(self.run_training(80.0)[1], 80.0)
        self.assertFalse(self.marker.exists())

    def test_over_80_seconds_fails_even_with_exit_zero(self):
        with self.assertRaisesRegex(TrainingRunLimitExceeded, 'base_1 exceeded 80 seconds'):
            self.run_training(80.001)
        record = json.loads(self.marker.read_text())
        self.assertGreater(record['elapsed_s'], 80)
        self.assertEqual(record['run_name'], 'base_1')
        self.common.kill_group.assert_called_once_with(self.process)

    def test_deadline_kills_process_and_records_failure(self):
        with self.assertRaises(TrainingRunLimitExceeded):
            self.run_training(80.01, error=True)
        self.assertTrue(json.loads(self.marker.read_text())['deadline_reached'])
        self.common.kill_group.assert_called_once_with(self.process)
        self.assertIn('TIMEOUT', self.path.read_text())

    def test_shorter_caller_timeout_is_preserved(self):
        self.assertEqual(self.run_training(5.2, timeout=5, error=True)[0], -9)
        self.assertEqual(self.process.wait.call_args.kwargs['timeout'], 5)
        self.assertFalse(self.marker.exists())

    def test_warmup_and_evaluation_keep_original_timeout(self):
        wrapped = wrap_run_logged(self.common)
        for command in (['python', '-c', 'print(1)'], ['python', '/repo/harness/eval_adapter.py']):
            self.assertEqual(wrapped(command, self.path, 900), (0, 100, 0, 0))
        self.assertEqual(self.original.call_count, 2)
        self.common.subprocess.Popen.assert_not_called()

    def test_progress_keeps_peak_and_elapsed_when_memory_check_raises(self):
        samples = []
        def fail(*args):
            raise RuntimeError('memory exceeded')
        wrapped = wrap_run_logged(self.common, memory_limit=lambda cmd:50,
            memory_exceeded=fail, on_progress=lambda *args:samples.append(args))
        with self.assertRaisesRegex(RuntimeError,'memory exceeded'):
            wrapped(['python','_train.py'],self.path,900)
        self.common.kill_group.assert_called_once_with(self.process)
        self.assertEqual(samples[-1][3],100)
        self.assertGreaterEqual(samples[-1][2],0)
        self.assertFalse(samples[-1][4])

    def test_progress_keeps_peak_and_elapsed_when_deadline_expires(self):
        samples = []
        values=iter([0,0.1,0.2,80.01,80.02,80.03])
        wrapped=wrap_run_logged(self.common,clock=lambda:next(values),failure_path=self.marker,
            on_progress=lambda *args:samples.append(args))
        with self.assertRaises(TrainingRunLimitExceeded):
            wrapped(['python','_train.py'],self.path,900)
        self.assertEqual(samples[-1][3],100)
        self.assertGreater(samples[-1][2],80)
        self.assertFalse(samples[-1][4])

    def test_each_repeat_gets_its_own_80_second_budget(self):
        values = iter([100, 100.2, 179, 200, 200.2, 279])
        wrapped = wrap_run_logged(self.common, clock=lambda: next(values), failure_path=self.marker)
        for name in ('base_1', 'solution_1'):
            path = self.root / name / 'train.log'
            path.parent.mkdir(exist_ok=True)
            self.assertEqual(wrapped(['python', '_train.py'], path, 900)[1], 79)
        self.assertFalse(self.marker.exists())

    def test_worker_reads_marker_even_if_notebook_handles_exception(self):
        delivery = self.root / 'output' / 'delivery'
        delivery.mkdir(parents=True)
        (delivery / 'training_limit.json').write_text(json.dumps({
            'reason': 'training_run_exceeded_limit', 'limit_s': 80,
            'elapsed_s': 80.001, 'deadline_reached': True, 'run_name': 'base_1'}))
        worker = object.__new__(Worker)
        self.assertIn('base_1 exceeded 80 seconds', worker.training_limit_failure(self.root))

    def test_host_watchdog_times_training_pid_without_startup_hook(self):
        process = self.root / '1234'
        process.mkdir()
        fields = ['S'] + ['0'] * 18 + [str(100 * os.sysconf('SC_CLK_TCK'))]
        (process / 'stat').write_text('1234 (python process) ' + ' '.join(fields))
        output = 'PID COMMAND\n1234 python /repo/harness/_train.py --config /runs/base_1/task.yaml\n'
        self.assertIsNone(training_process_deadline(output, process_root=self.root, now=179.16))
        self.assertIsNone(training_process_deadline(output, process_root=self.root, now=180))
        self.assertEqual(training_process_deadline(output, process_root=self.root, now=180.01)['run_name'], 'base_1')
        self.assertIsNone(training_process_deadline(output.replace('_train.py', 'eval_adapter.py'), process_root=self.root, now=300))


if __name__ == '__main__':
    unittest.main(verbosity=2)
