"""No-network tests for Sheet ingestion, durable retries and credential boundaries."""
import base64
import csv
import io
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import sheets_queue_worker as queue


NOTEBOOK = json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': {},
    'cells': [{'cell_type': 'code', 'source': 'raise AssertionError("must never execute on ingestion host")',
               'metadata': {}, 'outputs': [], 'execution_count': None}]})


def row(rid='example-1', **changes):
    result = {name: '' for name in queue.HEADERS}
    result.update({'Request ID': rid, 'Candidate ID': 'candidate-1', 'Notebook JSON': NOTEBOOK, 'Ready': 'TRUE'})
    result.update(changes)
    return result


def csv_bytes(rows, headers=queue.HEADERS):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


class FakeAPI:
    def __init__(self):
        self.jobs, self.calls = {}, []
        self.capacity_blocked = False
        self.lose_response_once = False

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == 'POST':
            if self.capacity_blocked:
                raise queue.RemoteError('Local API rejected request (HTTP 400)', True)
            rid = body['request_id']
            raw = base64.b64decode(body['notebook_base64'])
            old = self.jobs.get(rid)
            if old:
                assert old['source_sha'] == queue.digest(raw), 'Idempotency conflict'
                return dict(old)
            job = {'id': 'n_' + format(len(self.jobs) + 1, '032x'), 'request_id': rid,
                   'source_sha': queue.digest(raw), 'state': 'queued', 'candidate': body['candidate_id'],
                   'metrics': {'available': False}, 'result': None}
            self.jobs[rid] = job
            if self.lose_response_once:
                self.lose_response_once = False
                raise queue.RemoteError('Lost response after acceptance', True)
            return dict(job)
        return dict(next(job for job in self.jobs.values() if path.endswith(job['id'])))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.api = FakeAPI()
        self.rows = [row()]
        self.downloads = []
        self.sheet_unavailable = False
        self.worker = self.new_worker()

    def new_worker(self):
        def fetch(url, limit):
            self.downloads.append((url, limit))
            if url == queue.SHEET_URL:
                if self.sheet_unavailable:
                    raise queue.RemoteError('Public download unavailable')
                return csv_bytes(self.rows)
            return NOTEBOOK.encode()
        return queue.Worker(self.directory.name, self.api, fetch=fetch)

    def tearDown(self):
        self.worker.close()
        self.directory.cleanup()

    def restart(self):
        self.worker.close()
        self.worker = self.new_worker()

    def receipt(self, name='example-1.json'):
        return json.loads((Path(self.directory.name) / 'outbox' / name).read_text())

    def test_ready_row_creates_site_job_with_stable_id_and_raw_snapshot(self):
        counts = self.worker.tick()
        self.assertEqual((counts['accepted'], counts['submitted']), (1, 1))
        self.assertEqual(len(self.api.jobs), 1)
        method, path, body = self.api.calls[0]
        self.assertEqual((method, path, body['request_id']), ('POST', '/v1/notebooks', 'sheet-example-1'))
        self.assertFalse(body['contains_secrets'])
        self.assertEqual((body['timeout_s'], body['cell_timeout_s']), (3600, 1800))
        snapshot = Path(self.directory.name) / 'inputs' / 'example-1.ipynb'
        self.assertEqual(snapshot.read_bytes(), NOTEBOOK.encode())
        self.assertEqual(self.receipt()['source_sha256'], queue.digest(NOTEBOOK.encode()))
        self.assertEqual(self.receipt()['state'], 'queued')

    def test_only_true_rows_trigger_and_multiple_rows_are_independent(self):
        self.rows = [row(str(index), Ready=ready) for index, ready in enumerate(['TRUE', 'true', 'False', '', '1', 'yes'])]
        self.worker.tick()
        self.assertEqual(set(self.api.jobs), {'sheet-0', 'sheet-1'})

    def test_repeat_poll_and_restart_do_not_resubmit_existing_job(self):
        self.worker.tick()
        self.restart()
        self.worker.tick()
        self.worker.tick()
        self.assertEqual(sum(method == 'POST' for method, _, _ in self.api.calls), 1)
        self.assertEqual(len(self.api.jobs), 1)

    def test_lost_post_response_retries_identical_id_without_second_gpu_job(self):
        self.api.lose_response_once = True
        self.worker.tick()
        self.assertEqual(self.receipt()['state'], 'pending')
        self.assertEqual(len(self.api.jobs), 1)
        self.restart()
        self.worker.tick()
        self.assertEqual(len(self.api.jobs), 1)
        self.assertEqual(sum(method == 'POST' for method, _, _ in self.api.calls), 2)
        self.assertEqual(self.receipt()['state'], 'queued')

    def test_crash_between_post_and_ledger_update_recovers_without_duplicate(self):
        with patch.object(self.worker, 'save_job', side_effect=RuntimeError('simulated process crash')):
            with self.assertRaises(RuntimeError):
                self.worker.tick()
        self.assertEqual(len(self.api.jobs), 1)
        self.restart()
        self.worker.tick()
        self.assertEqual(len(self.api.jobs), 1)
        self.assertEqual(self.receipt()['state'], 'queued')

    def test_edited_accepted_id_is_conflict_and_new_id_runs(self):
        self.worker.tick()
        self.rows[0]['Candidate ID'] = 'changed-candidate'
        self.worker.tick()
        self.assertIn('REQUEST_ID_CONFLICT', self.receipt('example-1.conflict.json')['error'])
        self.assertEqual(len(self.api.jobs), 1)
        self.rows[0]['Request ID'] = 'example-2'
        self.worker.tick()
        self.assertEqual(len(self.api.jobs), 2)

    def test_duplicate_ready_ids_do_not_execute(self):
        self.rows = [row(), row()]
        self.worker.tick()
        self.assertEqual(self.api.jobs, {})
        self.assertIn('Duplicate', self.receipt('example-1.rejected.json')['error'])

    def test_invalid_row_is_reported_and_does_not_block_other_rows(self):
        self.rows = [row(**{'Notebook JSON': '<html>Access denied</html>'}), row('valid-2')]
        counts = self.worker.tick()
        self.assertEqual(counts['errors'], 1)
        self.assertEqual(set(self.api.jobs), {'sheet-valid-2'})
        self.assertEqual(self.receipt('example-1.rejected.json')['state'], 'rejected')
        self.assertEqual(self.worker.db.execute('SELECT count(*) FROM rejections').fetchone()[0], 1)

    def test_capacity_failure_remains_pending_then_retries(self):
        self.api.capacity_blocked = True
        self.worker.tick()
        self.assertEqual(self.receipt()['state'], 'pending')
        self.api.capacity_blocked = False
        self.worker.tick()
        self.assertEqual(self.receipt()['state'], 'queued')
        self.assertEqual(len(self.api.jobs), 1)

    def test_snapshot_tampering_prevents_submission(self):
        self.api.capacity_blocked = True
        self.worker.tick()
        (Path(self.directory.name) / 'inputs' / 'example-1.ipynb').write_text(NOTEBOOK + ' ')
        self.api.capacity_blocked = False
        self.worker.tick()
        self.assertEqual(self.api.jobs, {})
        self.assertEqual(self.receipt()['state'], 'rejected')
        self.assertIn('SOURCE_INTEGRITY_ERROR', self.receipt()['error'])

    def test_full_metrics_and_execution_error_are_retained(self):
        self.worker.tick()
        self.api.jobs['sheet-example-1'].update(state='failed', metrics={'training_s': 80, 'gpu_peak_mib': 14345, 'eval_loss': 2.05},
            result={'error': {'code': 'TIME_LIMIT_EXCEEDED', 'limit': 80}})
        self.worker.tick()
        receipt = self.receipt()
        self.assertEqual(receipt['state'], 'failed')
        self.assertEqual(receipt['job']['metrics']['gpu_peak_mib'], 14345)
        self.assertEqual(receipt['job']['result']['error']['code'], 'TIME_LIMIT_EXCEEDED')

    def test_sheet_outage_does_not_stop_existing_job_tracking(self):
        self.worker.tick()
        self.api.jobs['sheet-example-1']['state'] = 'completed'
        self.sheet_unavailable = True
        self.worker.tick()
        self.assertEqual(self.receipt()['state'], 'completed')
        self.assertEqual(self.receipt('sheet-status.json')['state'], 'unavailable')
        self.sheet_unavailable = False
        self.worker.tick()
        self.assertEqual(self.receipt('sheet-status.json')['state'], 'available')

    def test_public_notebook_download_is_snapshotted_only_once(self):
        self.rows = [row(**{'Notebook JSON': '', 'Notebook URL': 'https://example.invalid/redacted'})]
        self.worker.tick()
        self.restart()
        self.worker.tick()
        downloads = [url for url, _ in self.downloads if url != queue.SHEET_URL]
        self.assertEqual(downloads, ['https://drive.usercontent.google.com/download?id=file_123&export=download&confirm=t'])

    def test_inflight_snapshot_count_is_bounded_to_32(self):
        self.rows = [row('request-' + str(index)) for index in range(40)]
        self.api.capacity_blocked = True
        self.worker.tick()
        self.worker.tick()
        self.assertEqual(self.worker.db.execute('SELECT count(*) FROM entries').fetchone()[0], 32)
        self.assertEqual(len(list((Path(self.directory.name) / 'inputs').glob('*.ipynb'))), 32)

    def test_writeback_failure_does_not_block_submission_and_retries_full_result(self):
        class Publisher:
            blocked = True
            tables = None
            def publish(self, tables):
                self.tables = tables
                return {'state': 'retry' if self.blocked else 'published', 'error': 'Google Sheets HTTP 403'}
        publisher = Publisher()
        self.worker.publisher = publisher
        self.assertEqual(self.worker.tick()['results'], 'retrying')
        self.assertEqual(len(self.api.jobs), 1)
        self.api.jobs['sheet-example-1'].update(state='failed',
            metrics={'display': {'wall_s': 80, 'gpu_peak_mib': 14000, 'eval_loss': 2.1, 'partial': True}},
            result={'error': {'code': 'TIME_LIMIT_EXCEEDED', 'phase': 'training'}})
        publisher.blocked = False
        self.assertEqual(self.worker.tick()['results'], 'synchronized')
        table = publisher.tables['VM Results']
        record = dict(zip(table[0], table[1]))
        self.assertEqual((record['GPU peak (MiB)'], record['Eval loss'], record['Time check']), (14000, 2.1, 'failed'))
        self.assertEqual(sum(method == 'POST' for method, _, _ in self.api.calls), 1)

    def test_restart_backfills_completed_jobs_without_running_them_again(self):
        self.worker.tick()
        self.api.jobs['sheet-example-1']['state'] = 'completed'
        self.worker.tick()
        self.restart()
        class Publisher:
            tables = None
            def publish(self, tables):
                self.tables = tables
                return {'state': 'published'}
        publisher = Publisher()
        self.worker.publisher = publisher
        self.rows = []
        self.worker.tick()
        self.assertEqual(publisher.tables['VM Results'][1][3], 'completed')
        self.assertEqual(sum(method == 'POST' for method, _, _ in self.api.calls), 1)

    def test_unexpected_publish_exception_is_sanitized_and_intake_continues(self):
        class Publisher:
            def publish(self, tables):
                raise RuntimeError('private-credential-never-print')
        self.worker.publisher = Publisher()
        self.assertEqual(self.worker.tick()['results'], 'retrying')
        self.assertEqual(len(self.api.jobs), 1)
        self.assertNotIn('private-credential', json.dumps(self.receipt('publisher-status.json')))

    def test_control_and_error_receipt_names_cannot_collide_with_request_ids(self):
        self.rows = [row(rid) for rid in ('sheet-status', 'publisher-status', 'x.conflict', 'x.rejected', 'invalid-row-7')]
        self.worker.tick()
        self.assertEqual(len(self.api.jobs), 0)
        self.assertEqual(self.receipt('sheet-status.json')['state'], 'available')
        for number in range(2, 7):
            self.assertEqual(self.receipt('invalid-row-' + str(number) + '.json')['state'], 'rejected')

    def test_fixed_input_removes_stale_rejection_but_keeps_audit_and_job(self):
        self.rows = [row(**{'Notebook JSON': '<invalid>'})]
        self.worker.tick()
        self.assertEqual(self.receipt('example-1.rejected.json')['state'], 'rejected')
        self.rows = [row()]
        self.worker.tick()
        self.assertFalse((Path(self.directory.name) / 'outbox/example-1.rejected.json').exists())
        self.assertEqual(self.receipt()['state'], 'queued')
        self.assertEqual(self.worker.db.execute('SELECT count(*) FROM rejections').fetchone()[0], 1)


class ValidationTests(unittest.TestCase):
    def test_notebook_url_allowlist_and_normalization(self):
        self.assertEqual(queue.notebook_url('https://github.com/org/repo/blob/main/path/test.ipynb'),
            'https://raw.githubusercontent.com/org/repo/main/path/test.ipynb')
        self.assertEqual(queue.notebook_url('https://raw.githubusercontent.com/org/repo/main/test.ipynb'),
            'https://raw.githubusercontent.com/org/repo/main/test.ipynb')
        self.assertEqual(queue.notebook_url('https://example.invalid/redacted'),
            'https://drive.usercontent.google.com/download?id=abc&export=download&confirm=t&resourcekey=0-public-key')
        for url in ('https://example.invalid/redacted', 'https://127.0.0.1/test.ipynb',
                    'https://169.254.169.254/latest/meta-data', 'https://attacker.invalid/test.ipynb',
                    'https://example.invalid/redacted',
                    'https://example.invalid/redacted',
                    'https://github.com:8765/org/repo/blob/main/x.ipynb',
                    'https://github.com:invalid/org/repo/blob/main/x.ipynb',
                    'https://github.com/org/repo/blob/main/%2E%2E/x.ipynb',
                    'https://github.com/org/repo/blob/main/test.py',
                    'https://raw.githubusercontent.com/org/repo/main/x.ipynb?token=secret'):
            with self.subTest(url=url), self.assertRaises(queue.InvalidInput):
                queue.notebook_url(url)

    def test_download_redirect_rejects_arbitrary_hosts_and_http(self):
        redirect = queue.DownloadRedirect()
        request = urllib.request.Request('https://drive.usercontent.google.com/download?id=abc')
        for target in ('https://attacker.invalid/file', 'http://googleusercontent.com/file',
                       'https://127.0.0.1/private', 'https://example.invalid/redacted'):
            with self.subTest(target=target), self.assertRaises(queue.InvalidInput):
                redirect.redirect_request(request, None, 302, 'Moved', {}, target)
        result = redirect.redirect_request(request, None, 302, 'Moved', {}, 'https://doc-abc.googleusercontent.com/file')
        self.assertEqual(result.full_url, 'https://doc-abc.googleusercontent.com/file')

    def test_sheet_url_is_limited_to_csv_exports(self):
        self.assertEqual(queue.sheet_url(queue.SHEET_URL), queue.SHEET_URL)
        for url in ('https://attacker.invalid/export?format=csv&gid=1', queue.SHEET_URL + '&token=secret',
                    queue.SHEET_URL.replace('format=csv', 'format=xlsx')):
            with self.assertRaises(queue.InvalidInput):
                queue.sheet_url(url)

    def test_result_columns_cannot_expand_intake_export(self):
        legacy = queue.SHEET_URL.split('&range=')[0]
        self.assertEqual(queue.sheet_url(legacy), queue.SHEET_URL)
        for value in ('A1:BS1000', 'A1:H1001', 'A:H', ''):
            with self.subTest(value=value), self.assertRaises(queue.InvalidInput):
                queue.sheet_url(legacy + '&range=' + value)

    def test_output_columns_do_not_change_notebook_specification(self):
        original = row()
        expanded = dict(original, Status='completed', **{'GPU peak (MiB)': '0'})
        self.assertEqual(queue.specification(original), queue.specification(expanded))

    def test_invalid_ids_both_sources_and_timeout_limits(self):
        changes = [{'Request ID': '../outside'}, {'Request ID': 'x' * 81}, {'Candidate ID': 'has spaces'},
            {'Notebook URL': 'https://github.com/org/repo/blob/main/test.ipynb'},
            {'Notebook JSON': ''}, {'Timeout (s)': '9'}, {'Timeout (s)': '14401'},
            {'Timeout (s)': '80', 'Cell timeout (s)': '81'}, {'Timeout (s)': '-1'},
            {'Notebook JSON': 'x' * 50001}]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(queue.InvalidInput):
                queue.specification(row(**change))
        spec = queue.specification(row(**{'Timeout (s)': '80'}))
        self.assertEqual((spec['timeout_s'], spec['cell_timeout_s']), (80, 80))

    def test_csv_headers_rows_and_notebook_structure_are_bounded(self):
        with self.assertRaises(queue.InvalidInput):
            queue.parse_sheet(b'wrong,headers\n1,2\n')
        with self.assertRaises(queue.InvalidInput):
            queue.parse_sheet(csv_bytes([row(str(index)) for index in range(1001)]))
        for raw in (b'<html>login</html>', b'[]', b'{"nbformat":4,"cells":[]}',
                    b'{"nbformat":4,"cells":[{"cell_type":"markdown","source":"text"}]}'):
            with self.assertRaises(queue.InvalidInput):
                queue.validate_notebook(raw)

    def test_local_api_only_uses_fixed_loopback_and_never_follows_redirect(self):
        token = 'private_' + 'x' * 40
        client = queue.SoupAPI(token)
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return b'{"ok":true}'
        captured = []
        def open_request(request, timeout):
            captured.append(request)
            return Reply()
        with patch.object(client.opener, 'open', side_effect=open_request):
            self.assertEqual(client.request('POST', '/v1/notebooks', {'example': 1}), {'ok': True})
        self.assertEqual(captured[0].full_url, 'http://127.0.0.1:8765/v1/notebooks')
        self.assertEqual(captured[0].get_header('Authorization'), 'Bearer ' + token)
        with self.assertRaises(queue.InvalidInput):
            client.request('GET', 'https://attacker.invalid/')
        with self.assertRaises(queue.RemoteError):
            queue.NoRedirect().redirect_request(None, None, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
