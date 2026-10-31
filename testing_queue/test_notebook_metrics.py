import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from notebook_metrics import summarize_archive, comparison
from notebook_queue import NotebookQueue
from test_notebooks import CONFIG, submission


def report(base_time=35, solution_time=30, base_memory=2000, solution_memory=1800,
           base_loss=2, solution_loss=2.001, packages=None):
    return {'summary': {
        'base': {'ok': True, 'wall_s': base_time, 'gpu_peak_mib': base_memory, 'eval_loss': base_loss, 'runs': 1},
        'solution': {'ok': True, 'wall_s': solution_time, 'gpu_peak_mib': solution_memory, 'eval_loss': solution_loss, 'runs': 1}},
        'packages_changed': [] if packages is None else packages}


def bundle(value):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('execution.log', 'test')
        z.writestr('runner.json', '{}')
        z.writestr('results/result.json', json.dumps(value))
    return stream.getvalue()


class MetricsTests(unittest.TestCase):
    def partial(self, files, reference=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            for name, value in files.items():
                z.writestr(name, json.dumps(value))
        return summarize_archive(io.BytesIO(stream.getvalue()), reference)

    def test_solution_only_reuses_compatible_saved_baseline(self):
        reference = {'task_fingerprint':'task', 'image_id':'image', 'eval_data_sha256':'eval',
                     'job_id':'saved', 'metrics':report()['summary']['base']}
        data = report()
        del data['summary']['base']
        files = {'results/result.json':data, 'runner.json':{'image_id':'image'},
                 'results/run-metrics.json':{'runs':[{'run':'solution_1','side':'solution',
                    'task_fingerprint':'task','eval_data_sha256':'eval','reference_compatible':True}]}}
        m = self.partial(files, reference)
        self.assertEqual(m['baseline_source'],'saved')
        self.assertAlmostEqual(m['speedup'],35/30)
        self.assertEqual(comparison(m,'completed'),'beats_baseline')
        for key in ('image_id','task_fingerprint','eval_data_sha256'):
            bad = {**reference,key:'different'}
            self.assertIsNone(self.partial(files,bad)['speedup'])

    def test_timeout_preserves_peak_time_and_completed_loss(self):
        m = self.partial({'results/run-metrics.json':{'runs':[{'run':'solution_1','side':'solution',
            'wall_s':80.002,'gpu_peak_mib':14345,'eval_loss':2.1,'partial':True}]},
            'results/training-limit.json':{'run_name':'solution_1','elapsed_s':80.003}})
        self.assertEqual(m['display']['wall_s'],80.003)
        self.assertEqual(m['display']['gpu_peak_mib'],14345)
        self.assertEqual(m['display']['eval_loss'],2.1)
        self.assertEqual(m['speedup_kind'],'upper_bound')
        self.assertEqual(comparison(m,'failed'),'run_failed')

    def test_old_failure_and_eval_outputs_are_backfilled(self):
        m = self.partial({'results/training-limit.json':{'run_name':'base_1','elapsed_s':80.01},
            'results/harness/run-eval_solution.log.json':{'mean_loss':2.02},
            'results/failure.json':{'code':'EVAL_LOSS_EXCEEDED','measured':2.02,'baseline':2}})
        self.assertEqual(m['base']['wall_s'],80.01)
        self.assertEqual(m['solution']['eval_loss'],2.02)
        self.assertEqual(m['checks']['loss'],'failed')

    def test_host_peak_survives_kernel_death_without_fabricated_training_or_loss(self):
        m = self.partial({'results/host-metrics.json':{'gpu_peak_mib':15000}})
        self.assertEqual(m['display']['gpu_peak_mib'],15000)
        self.assertEqual(m['display']['gpu_scope'],'whole_notebook')
        self.assertIsNone(m['display']['wall_s'])
        self.assertEqual(m['display']['eval_status'],'not_evaluated')

    def test_malformed_partial_reports_do_not_break_job_listing(self):
        for failure in ('bad', {'phase':'training','run':['bad']}):
            m = self.partial({'results/failure.json':failure,'results/run-metrics.json':{
                'runs':[{'run':'solution_1','wall_s':math.inf,'gpu_peak_mib':math.nan,'extra':math.inf}]}})
            json.dumps(m,allow_nan=False)
            self.assertIsNone(m['solution']['wall_s'])

    def summarize(self, value):
        return summarize_archive(io.BytesIO(bundle(value)))

    def test_faster_lower_memory_and_valid_loss_beats_base(self):
        m = self.summarize(report())
        self.assertAlmostEqual(m['speedup'], 35/30)
        self.assertAlmostEqual(m['memory_saved_pct'], 100*(1-1800/2000))
        self.assertEqual(m['checks'], {'loss':'passed','time':'passed','memory':'passed','environment':'unchanged'})
        self.assertEqual(comparison(m, 'completed'), 'beats_baseline')

    def test_exact_hard_limits_pass_but_unrounded_excess_fails(self):
        at_limit = self.summarize(report(base_time=38.49, solution_time=38.49,
                                        base_memory=2261, solution_memory=2261))
        self.assertEqual(at_limit['checks']['time'], 'passed')
        self.assertEqual(at_limit['checks']['memory'], 'passed')
        over_time = self.summarize(report(solution_time=38.49001))
        self.assertEqual(over_time['checks']['time'], 'failed')
        over_memory = self.summarize(report(solution_memory=2262))
        self.assertEqual(over_memory['checks']['memory'], 'failed')
        no_base = report(solution_memory=2262)
        del no_base['summary']['base']
        self.assertEqual(self.summarize(no_base)['checks']['memory'], 'failed')

    def test_tradeoffs_are_not_presented_as_unqualified_wins(self):
        self.assertEqual(comparison(self.summarize(report(solution_memory=2100)), 'completed'), 'faster_more_memory')
        self.assertEqual(comparison(self.summarize(report(solution_time=38)), 'completed'), 'less_memory')
        self.assertEqual(comparison(self.summarize(report(solution_time=38, solution_memory=2100)), 'completed'), 'no_gain')

    def test_loss_time_memory_and_environment_failures_override_speed(self):
        for changes in ({'solution_loss':2.02}, {'base_time':38.491}, {'solution_memory':2262},
                        {'packages':['torch==different']}):
            with self.subTest(changes=changes):
                self.assertEqual(comparison(self.summarize(report(**changes)), 'completed'), 'checks_failed')

    def test_no_matching_base_is_unknown_not_a_win(self):
        data=report()
        del data['summary']['base']
        m=self.summarize(data)
        self.assertEqual(m['checks']['loss'],'unknown')
        self.assertEqual(comparison(m,'completed'),'incomplete')
        self.assertIsNone(m['speedup'])

    def test_failed_or_active_job_never_beats_even_if_metrics_claim_success(self):
        m=self.summarize(report())
        for state in ('failed','timed_out','interrupted','cancelled'):
            self.assertEqual(comparison(m,state),'run_failed')
        for state in ('queued','running'):
            self.assertEqual(comparison(m,state),'pending')

    def test_nonfinite_boolean_negative_metrics_cannot_be_wins(self):
        for value in (math.nan, math.inf, -1, True, '60', None):
            m=self.summarize(report(solution_time=value))
            self.assertIsNone(m['solution']['wall_s'])
            self.assertEqual(comparison(m,'completed'),'incomplete')
            json.dumps(m,allow_nan=False)

    def test_missing_environment_evidence_stays_incomplete(self):
        data=report()
        del data['packages_changed']
        self.assertEqual(comparison(self.summarize(data),'completed'),'incomplete')

    def test_missing_reports_and_oversized_reports_are_bounded(self):
        self.assertFalse(self.summarize({})['available'])
        huge={'padding':'x'*(2*1024**2),**report()}
        self.assertFalse(self.summarize(huge)['available'])

    def test_existing_job_gets_cached_metrics_without_rerunning(self):
        with tempfile.TemporaryDirectory() as d:
            queue=NotebookQueue(Path(d),CONFIG)
            job=queue.submit(submission(),'api-submitter')
            queue.claim('api-worker')
            queue.upload(job['id'],'api-worker',bundle(report()))
            done=queue.finish(job['id'],'api-worker',{'state':'completed','exit_code':0,'elapsed_s':120,
                'container_cleanup_confirmed':True,'reason':'fixture','image_id':CONFIG['image_id']})
            self.assertEqual(done['metrics']['comparison'],'beats_baseline')
            cache=queue.root/job['id']/'metrics-summary.json'
            self.assertTrue(cache.is_file())
            before=cache.stat().st_mtime_ns
            restarted=NotebookQueue(Path(d),CONFIG)
            self.assertEqual(restarted.status(job['id'])['metrics'],done['metrics'])
            self.assertEqual(cache.stat().st_mtime_ns,before)


if __name__ == '__main__':
    unittest.main()

