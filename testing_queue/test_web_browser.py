"""Optional actual Chromium smoke: requires Playwright + Chromium, never GPU/network services."""
import json, shutil, tempfile, threading, time, unittest
from pathlib import Path
from api import LocalServer
from agent_client import Client
from playwright.sync_api import sync_playwright, expect
P=Path(__file__).resolve().parent
TOKENS={role:'TEST_ONLY_BROWSER_'+role.upper()+'_'+'x'*32 for role in ('submitter','reviewer','worker','viewer')}
class BrowserTests(unittest.TestCase):
 def test_actual_browser_upload_roles_clear_and_stale_responses(self):
  with tempfile.TemporaryDirectory() as tmp:
   server=LocalServer(Path(tmp)/'state',TOKENS);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    url='http://127.0.0.1:'+str(server.server_port)
    Client(url,TOKENS['reviewer']).request('POST','/v1/baselines',json.loads((P/'examples/baseline_cpu.json').read_text()))
    with sync_playwright() as pw:
     browser=pw.chromium.launch(executable_path=shutil.which('chromium'),headless=True,args=['--no-sandbox'])
     page=browser.new_page(viewport={'width':1280,'height':900});errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
     page.goto(url);expect(page.locator('h1')).to_have_text('Test queue');expect(page.locator('#submit')).to_be_disabled()
     page.locator('#token').fill('invalid');page.locator('#connect').click();expect(page.locator('#message')).to_contain_text('Invalid credentials')
     page.locator('#token').fill(TOKENS['viewer']);page.locator('#connect').click();expect(page.locator('#identity')).to_contain_text('viewer');expect(page.locator('#submit')).to_be_disabled()
     page.locator('#token').fill(TOKENS['submitter']);page.locator('#connect').click();expect(page.locator('#submit')).to_be_enabled()
     page.locator('#candidate').fill('BROWSER-SMOKE');page.locator('#baseline').fill('base_CPU');page.locator('#artifact').set_input_files(str(P/'examples/candidate.ipynb'));page.locator('#no-secrets').check();page.locator('#submit').click()
     expect(page.locator('#message')).to_contain_text('Submitted q_');expect(page.locator('#jobs tr')).to_have_count(1)
     page.locator('#jobs tr').click();expect(page.locator('#detail')).to_contain_text('source_sha256')
     page.locator('#submit').click();expect(page.locator('#message')).to_contain_text('Existing identical job returned');expect(page.locator('#jobs tr')).to_have_count(1)
     self.assertEqual(page.evaluate('Object.keys(localStorage).length + Object.keys(sessionStorage).length'),0)

     held=[]
     def intercept(route):held.append((route,route.fetch()))
     page.route('**/v1/jobs',intercept);page.locator('#refresh').click()
     deadline=time.monotonic()+10
     while not held:
      if time.monotonic()>deadline:raise AssertionError('Expected local request did not arrive')
      page.wait_for_timeout(10)
     page.locator('#clear').click();held[0][0].fulfill(response=held[0][1]);page.wait_for_timeout(100)
     expect(page.locator('#jobs tr')).to_have_count(0);expect(page.locator('#token')).to_have_value('');expect(page.locator('#submit')).to_be_disabled();expect(page.locator('#identity')).to_contain_text('Disconnected')
     page.unroute('**/v1/jobs',intercept)

     held=[];page.route('**/v1/session',intercept);page.locator('#token').fill(TOKENS['submitter']);page.locator('#connect').click()
     deadline=time.monotonic()+10
     while not held:
      if time.monotonic()>deadline:raise AssertionError('Expected local request did not arrive')
      page.wait_for_timeout(10)
     page.locator('#clear').click();held[0][0].fulfill(response=held[0][1]);page.wait_for_timeout(100)
     expect(page.locator('#submit')).to_be_disabled();expect(page.locator('#identity')).to_contain_text('Disconnected');expect(page.locator('#jobs tr')).to_have_count(0)
     self.assertEqual(errors,[]);browser.close()
   finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main(verbosity=2)
