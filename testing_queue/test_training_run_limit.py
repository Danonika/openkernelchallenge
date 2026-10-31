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
        values = iter([0.0, 0.2, elapsed])
        wrapped = wrap_run_logged(self.common, clock=lambda: next(values), failure_path=self.marker)
        if error:
            self.process.wait.side_effect = subprocess.TimeoutExpired('train', 38.49)
        return wrapped(['python', '/repo/harness/_train.py'], self.path, timeout)

    def test_38_16_seconds_passes(self):
        self.assertEqual(self.run_training(38.16), (0, 38.16, 100, 0))
        self.assertAlmostEqual(self.process.wait.call_args.kwargs['timeout'], 38.29)
        self.assertFalse(self.marker.exists())
        self.common.kill_group.assert_called_once_with(self.process)

    def test_exactly_38_49_seconds_passes(self):
        self.assertEqual(self.run_training(38.49)[1], 38.49)
        self.assertFalse(self.marker.exists())

    def test_over_38_49_seconds_fails_even_with_exit_zero(self):
        with self.assertRaisesRegex(TrainingRunLimitExceeded, 'base_1 exceeded 38.49 seconds'):
            self.run_training(38.491)
        record = json.loads(self.marker.read_text())
        self.assertGreater(record['elapsed_s'], 38.49)
        self.assertEqual(record['run_name'], 'base_1')
        self.common.kill_group.assert_called_once_with(self.process)

    def test_deadline_kills_process_and_records_failure(self):
        with self.assertRaises(TrainingRunLimitExceeded):
            self.run_training(38.50, error=True)
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
        values=iter([0,0.1,0.2,38.50,38.51,38.52])
        wrapped=wrap_run_logged(self.common,clock=lambda:next(values),failure_path=self.marker,
            on_progress=lambda *args:samples.append(args))
        with self.assertRaises(TrainingRunLimitExceeded):
            wrapped(['python','_train.py'],self.path,900)
        self.assertEqual(samples[-1][3],100)
        self…5488 tokens truncated…led_screen_exposed_in_api_result(self):
        job = self.queue.submit(submission(), 'api-submitter')
        self.queue.claim('api-worker')
        self.queue.upload(job['id'], 'api-worker', outputs())
        result = {'state': 'failed', 'exit_code': 1, 'elapsed_s': 80.1,
                  'container_cleanup_confirmed': True, 'reason': 'Training timeout',
                  'image_id': CONFIG['image_id'],
                  'error': error('TIME_LIMIT_EXCEEDED', 'Training timeout', 'training', limit=80)}
        done = self.queue.finish(job['id'], 'api-worker', result)
        self.assertEqual(done['result']['error']['code'], 'TIME_LIMIT_EXCEEDED')


if __name__ == '__main__':
    unittest.main()

