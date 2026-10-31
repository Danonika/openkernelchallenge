import io
import json
import math
import subprocess
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from notebook_failfast import (ScreeningPolicy, ScreeningFailed, classify_exception,
                               compare_loss, error, valid_error, container_failure)
from training_run_limit import wrap_run_logged
from notebook_failfast import gpu_memory_failure
from notebook_reference import load_reference
from notebook_worker import Worker
from test_notebooks import NotebookTests, CONFIG, outputs, submission


class FailFastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.failure = self.root / 'failure.json'
        self.log = self.root / 'train.log'
        self.log.write_text('')
        self.common = types.SimpleNamespace(run_logged=Mock(return_value=(0, 1, 100, 0)))
        self.policy = ScreeningPolicy(self.common, lambda common, **kw: common.run_logged,
                                      failure_path=self.failure)

    def tearDown(self):
        self.temp.cleanup()

    def test_strict_gpu_boundary_and_error_details(self):
        for peak in (None, 0, 2260, 2261):
            self.assertIsNone(gpu_memory_failure(peak))
        for peak in (2261.001, 2262):
            detail = gpu_memory_failure(peak, 'container')
            self.assertEqual(detail['code'], 'MEMORY_LIMIT_EXCEEDED')
            self.assertEqual(detail['limit'], 2261)
            self.assertEqual(detail['measured'], peak)
            self.assertEqual(detail['unit'], 'MiB')
            self.assertTrue(valid_error(detail))
        self.assertEqual(self.policy.memory_limit(['python', '_train.py']), 2261)
        self.assertEqual(self.policy.memory_limit(['python', '_train.py', '--solution']), 2261)

    def test_new_performance_reference_keeps_original_quality_reference(self):
        reference = load_reference()
        self.assertEqual(reference['metrics']['wall_s'], 38.49)
        self.assertEqual(reference['metrics']['gpu_peak_mib'], 2261)
        self.assertEqual(reference['metrics']['eval_loss'], 2.04968)
        self.assertEqual(reference['eval_reference']['job_id'], 'n_6e7375b093154fca990312c240e88402')
        self.assertEqual(reference['performance_reference']['job_id'], 'n_89db8d59aa5f4666b8f9c09f906481a2')
        self.assertIsNone(compare_loss(2.04968 * 1.005, reference['metrics']['eval_loss']))
        self.assertEqual(compare_loss(2.04968 * 1.005 + 0.000001, reference['metrics']['eval_loss'])['code'], 'EVAL_LOSS_EXCEEDED')

    def test_relative_loss_boundary_and_nonfinite_results(self):
        self.assertIsNone(compare_loss(2 * 1.005, 2))
        self.assertEqual(compare_loss(2.010001, 2)['code'], 'EVAL_LOSS_EXCEEDED')
        for loss in (None, '2', True, -1, math.nan, math.inf):
            self.assertEqual(compare_loss(loss, 2)['code'], 'INVALID_EVAL_RESULT')
        self.assertIsNone(compare_loss(0, 0))

    def test_allocation_errors_are_distinct_from_unknown_kills(self):
        for text in ('torch.OutOfMemoryError: CUDA out of memory',
                     'MemoryError: cannot allocate', 'DefaultCPUAllocator: not enough memory',
                     'std::bad_alloc'):
            self.assertEqual(classify_exception(text)['code'], 'MEMORY_LIMIT_EXCEEDED')
        self.assertEqual(classify_exception('Killed (137)')['code'], 'EXECUTION_ERROR')

    def test_container_oom_wins_over_consequent_kernel_death(self):
        kernel_error = error('EXECUTION_ERROR', 'Kernel died', 'notebook')
        detail = container_failure({'Running': True, 'OOMKilled': True}, kernel_error)
        self.assertEqual(detail['code'], 'MEMORY_LIMIT_EXCEEDED')
        self.assertEqual(detail['phase'], 'container')
        self.assertEqual(container_failure({'OOMKilled': False}, kernel_error), kernel_error)

    def test_storage_exhaustion_has_specific_error(self):
        for text in ('OSError: [Errno 28] No space left on device', 'Disk quota exceeded', 'ENOSPC'):
            self.assertEqual(classify_exception(text)['code'], 'STORAGE_LIMIT_EXCEEDED')

    def test_nonzero_training_stops_on_oom_and_preserves_log(self):
        self.common.run_logged.return_value = (1, 3, 15360, 0)
        self.log.write_text('torch.OutOfMemoryError: CUDA out of memory')
        with self.assertRaisesRegex(ScreeningFailed, 'MEMORY_LIMIT_EXCEEDED'):
            self.policy.run_logged(['python', '_train.py'], self.log, 900)
        self.assertEqual(json.loads(self.failure.read_text())['phase'], 'training')
        self.assertTrue(any((self.root / 'harness').glob('*.log')))

    def evaluate(self, adapter, loss, data='eval-a'):
        path = self.root / 'eval.json'
        path.write_text(json.dumps({'mean_loss': loss}))
        eval_file = self.root / 'eval.jsonl'
        eval_file.write_text(data)
        return self.policy.run_logged(['python', 'eval_adapter.py', '--adapter', adapter,
            '--out', str(path), '--eval-file', str(eval_file)], self.log, 900)

    def register_pair(self):
        for adapter, side in (('/adapter/base', 'base'), ('/adapter/solution', 'solution')):
            self.policy.records[str(Path(adapter).resolve())] = {
                'fingerprint': 'same-task', 'side': side, 'peak': 100}

    def test_paired_evaluation_stops_at_full_bad_loss(self):
        self.register_pair()
        self.evaluate('/adapter/base', 2)
        with self.assertRaisesRegex(ScreeningFailed, 'EVAL_LOSS_EXCEEDED'):
            self.evaluate('/adapter/solution', 2.02)
        detail = json.loads(self.failure.read_text())
        self.assertEqual(detail['baseline'], 2)
        self.assertAlmostEqual(detail['limit'], 2.01)
        self.assertEqual(detail['measured'], 2.02)

    def test_good_pair_emits_screening_evidence(self):
        self.register_pair()
        self.evaluate('/adapter/base', 2)
        self.evaluate('/adapter/solution', 2.001)
        self.assertEqual(json.loads((self.root / 'screening.json').read_text())['eval_loss_check'], 'passed')
        self.assertFalse(self.failure.exists())

    def test_changed_eval_rows_or_task_never_reuses_base_loss(self):
        for change in ('data', 'task'):
            self.register_pair()
            self.evaluate('/adapter/base', 2)
            if change == 'task':
                self.policy.records[str(Path('/adapter/solution').resolve())]['fingerprint'] = 'changed'
            with self.assertRaisesRegex(ScreeningFailed, 'BASELINE_REQUIRED'):
                self.evaluate('/adapter/solution', 1, data='different' if change == 'data' else 'eval-a')
            self.policy.evals.clear()

    def test_solution_only_evaluation_requires_baseline(self):
        self.register_pair()
        with self.assertRaisesRegex(ScreeningFailed, 'BASELINE_REQUIRED'):
            self.evaluate('/adapter/solution', 2)

    def test_solution_only_uses_saved_reference_and_keeps_bad_loss(self):
        import hashlib
        self.register_pair()
        self.policy.reference = {'task_fingerprint':'same-task',
            'eval_data_sha256':hashlib.sha256(b'eval-a').hexdigest(),
            'metrics':{'eval_loss':2,'gpu_peak_mib':100}, 'job_id':'saved'}
        self.evaluate('/adapter/solution',2.001)
        with self.assertRaisesRegex(ScreeningFailed,'EVAL_LOSS_EXCEEDED'):
            self.evaluate('/adapter/solution',2.02)
        record=json.loads((self.root/'run-metrics.json').read_text())['runs'][0]
        self.assertEqual(record['eval_loss'],2.02)
        self.assertEqual(json.loads((self.root/'screening.json').read_text())['eval_loss_check'],'failed')

    def test_missing_and_nan_eval_fail(self):
        self.register_pair()
        with self.assertRaisesRegex(ScreeningFailed, 'INVALID_EVAL_RESULT'):
            self.evaluate('/adapter/base', float('nan'))

    def test_first_failure_cannot_be_replaced_by_cell_wrapper_error(self):
        for code in ('EVAL_LOSS_EXCEEDED', 'EXECUTION_ERROR'):
            with self.assertRaises(ScreeningFailed):
                self.policy.fail(error(code, 'failed', 'evaluation'))
        self.assertEqual(json.loads(self.failure.read_text())['code'], 'EVAL_LOSS_EXCEEDED')

    def test_memory_watchdog_kills_process_group_before_waiting(self):
        class Poller:
            peak, start = 111, 0
            def __enter__(self): return self
            def __exit__(self, *args): return False
        proc = Mock()
        common = types.SimpleNamespace(run_logged=Mock(), GpuPoller=Poller,
            subprocess=types.SimpleNamespace(Popen=Mock(return_value=proc),
                STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
            REPO=self.root, ENV={}, kill_group=Mock())
        wrapped = wrap_run_logged(common, memory_limit=lambda cmd: 110,
                                  memory_exceeded=self.policy.memory_exceeded)
        with self.assertRaisesRegex(ScreeningFailed, 'MEMORY_LIMIT_EXCEEDED'):
            wrapped(['python', '_train.py'], self.log, 900)
        common.kill_group.assert_called_once_with(proc)
        proc.wait.assert_not_called()

    def test_error_schema_rejects_nonfinite_or_unknown_claims(self):
        self.assertTrue(valid_error(error('TIME_LIMIT_EXCEEDED', 'failed', 'training', limit=80)))
        for value in ({'code': 'PASS'}, error('TIME_LIMIT_EXCEEDED', 'failed', 'training', measured=math.nan),
                      error('EXECUTION_ERROR', 'failed', 'cell', secret='anything')):
            self.assertFalse(valid_error(value))

    def test_worker_reads_failure_and_collects_partial_harness_logs(self):
        delivery = self.root / 'output' / 'delivery'
        (delivery / 'harness').mkdir(parents=True)
        detail = error('EVAL_LOSS_EXCEEDED', 'Too large', 'evaluation', measured=2.1, limit=2.01)
        (delivery / 'failure.json').write_text(json.dumps(detail))
        (delivery / 'harness' / 'base_1-train.log').write_text('partial evidence')
        worker = object.__new__(Worker)
        self.assertEqual(worker.screening_failure(self.root), detail)
        with zipfile.ZipFile(io.BytesIO(worker.collect(self.root, b'log', {'error': detail}))) as archive:
            self.assertIn('results/failure.json', archive.namelist())
            self.assertIn('results/harness/base_1-train.log', archive.namelist())


class StructuredResultTests(NotebookTests):
    def test_failed_screen_exposed_in_api_result(self):
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

