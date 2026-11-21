"""Actual Chromium UI test with ephemeral state and public dummy tokens."""
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from api import LocalServer
from test_notebooks import TOKENS, CONFIG, notebook, outputs
from test_notebook_metrics import report, bundle as metric_bundle


class NotebookBrowserTests(unittest.TestCase):
    def test_submit_download_permissions_and_disconnect_races(self):
        with tempfile.TemporaryDirectory() as directory:
            server = LocalServer(Path(directory) / 'state', TOKENS, notebook_config=CONFIG)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(executable_path=shutil.which('chromium'), headless=True, args=['--no-sandbox'])
                    page = browser.new_page(viewport={'width': 1200, 'height': 1000})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto('http://127.0.0.1:' + str(server.server_port))
                    expect(page.locator('h1')).to_have_text('GPU notebook runner')
                    submit = page.locator('#submit button[type=submit]')
                    expect(submit).to_be_disabled()
                    page.locator('#token').fill(TOKENS['viewer'])
                    page.locator('#connect').click()
                    expect(page.locator('#status')).to_have_text('Connected')
                    expect(submit).to_be_disabled()
                    page.locator('#token').fill(TOKENS['submitter'])
                    page.locator('#connect').click()
                    expect(submit).to_be_enabled()
                    page.locator('#file').set_input_files({'name': 'test.ipynb', 'mimeType': 'application/json', 'buffer': notebook()})
                    page.locator('#no-secrets').check()
                    submit.click()
                    expect(page.locator('#status')).to_contain_text('Submitted n_')
                    expect(page.locator('#jobs .job-row')).to_have_count(1)
                    job = server.notebooks.claim('api-worker')
                    bundle = outputs()
                    server.notebooks.upload(job['id'], 'api-worker', bundle)
                    server.notebooks.finish(job['id'], 'api-worker', {'state': 'completed', 'exit_code': 0,
                        'elapsed_s': 1, 'reason': 'UI fixture', 'container_cleanup_confirmed': True, 'image_id': CONFIG['image_id']})
                    page.locator('#refresh').click()
                    expect(page.get_by_role('button', name='Outputs')).to_be_visible()
                    with page.expect_download() as download:
                        page.get_by_role('button', name='Outputs').click()
                    self.assertEqual(Path(download.value.path()).read_bytes(), bundle)
                    page.get_by_role('button', name='Re-run', exact=True).click()
                    expect(page.locator('#status')).to_contain_text('Re-run queued:')
                    expect(page.locator('#jobs .job-row')).to_have_count(2)
                    rerun = server.notebooks.status()[0]
                    self.assertEqual(rerun['rerun_of'], job['id'])
                    self.assertEqual(rerun['source_sha'], job['source_sha'])
                    server.notebooks.cancel(rerun['id'])
                    for name, value in (('fast-kernel', report()), ('memory-tradeoff', report(solution_time=75)),
                                        ('bad-quality', report(solution_loss=2.1))):
                        from test_notebooks import submission
                        created = server.notebooks.submit({**submission(name), 'candidate_id': name}, 'api-submitter')
                        server.notebooks.claim('api-worker')
                        server.notebooks.upload(created['id'], 'api-worker', metric_bundle(value))
                        server.notebooks.finish(created['id'], 'api-worker', {'state':'completed', 'exit_code':0,
                            'elapsed_s':120, 'reason':'Comparison fixture', 'container_cleanup_confirmed':True,
                            'image_id':CONFIG['image_id']})
                    page.locator('#refresh').click()
                    expect(page.locator('#jobs .job-row')).to_have_count(5)
                    expect(page.locator('#stat-wins')).to_have_text('1')
                    expect(page.locator('#stat-speedup')).to_have_text('1.17×')
                    page.locator('#sort').select_option('speedup')
                    page.locator('#filter').select_option('wins')
                    expect(page.locator('#jobs .job-row')).to_have_count(1)
                    expect(page.locator('#jobs .job-row')).to_contain_text('fast-kernel')
                    expect(page.locator('#jobs .job-row')).to_contain_text('Beat baseline')
                    page.get_by_role('button', name='Details', exact=True).click()
                    expect(page.locator('.detail-row')).to_contain_text('Matching baseline comparison')
                    page.locator('#refresh').click()
                    expect(page.locator('.detail-row')).to_have_count(1)
                    page.locator('#filter').select_option('all')
                    page.locator('#search').fill('bad-quality')
                    expect(page.locator('#jobs .job-row')).to_have_count(1)
                    expect(page.locator('#jobs .job-row')).to_contain_text('Checks failed')
                    page.locator('#search').fill('')
                    page.screenshot(path='/redacted/local/soup-queue-evidence/notebook-metrics-desktop.png', full_page=True)
                    page.set_viewport_size({'width':390,'height':844})
                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
                    page.screenshot(path='/redacted/local/soup-queue-evidence/notebook-metrics-mobile.png', full_page=True)
                    page.set_viewport_size({'width':1200,'height':1000})
                    self.assertEqual(page.evaluate('Object.keys(localStorage).length + Object.keys(sessionStorage).length'), 0)
                    held = []
                    def hold(route):
                        held.append((route, route.fetch()))
                    page.route('**/v1/notebooks', hold)
                    page.locator('#refresh').click()
                    deadline = time.monotonic() + 10
                    while not held:
                        assert time.monotonic() < deadline
                        page.wait_for_timeout(10)
                    page.locator('#disconnect').click()
                    held[0][0].fulfill(response=held[0][1])
                    page.wait_for_timeout(100)
                    expect(page.locator('#jobs .job-row')).to_have_count(0)
                    expect(page.locator('#status')).to_have_text('Disconnected')
                    expect(submit).to_be_disabled()
                    self.assertEqual(errors, [])
                    page.screenshot(path='/redacted/local/soup-queue-evidence/notebook-ui.png', full_page=True)
                    browser.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)

