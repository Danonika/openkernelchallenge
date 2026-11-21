"""Fake-runner infrastructure unit tests; no scientific/GPU/notebook execution."""
import copy,hashlib,io,json,os,sqlite3,subprocess,sys,tempfile,threading,unittest,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from soup_queue import Queue,QueueError
class Clock:
 def __init__(self):self.now=1000.
 def __call__(self):return self.now
class QueueTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'queue';self.clock=Clock();self.q=Queue(self.root,self.clock)
  for backend in ('CPU','KAGGLE_T4','GCP'):self.q.register_baseline(self.baseline(backend))
  self.stage()
 def tearDown(self):self.temp.cleanup()
 def baseline(self,b):return {'baseline_id':'base_'+b,'backend':b,'environment_fingerprint':'a'*64,'task_fingerprint':'b'*64,'harness_commit':'c'*40,'evidence_sha256':'d'*64}
 def stage(self,name='candidate.zip',members=None):
  with zipfile.ZipFile(self.root/'incoming'/name,'w') as z:
   for path,body in (members or {'solution/solution.py':'def apply():\n    return None\n'}).items():z.writestr(path,body)
 def spec(self,candidate='SOUP-NEW-001',backend='CPU',priority=50):return {'candidate_id':candidate,'backend':backend,'baseline_id':'base_'+backend,'entrypoint':'cpu_contract_v1' if backend=='CPU' else 'soup_paired_v1','artifact':'candidate.zip','artifact_kind':'solution_bundle','priority':priority,'submitted_by':'agentA','contains_secrets':False}
 def job(self,**kw):return self.q.submit(self.spec(**kw))
 def approve(self,j):self.q.review(j['job_id'],reviewer='jury',source_sha256=j['source_sha256'],approved=True,authorization_ref='parent-approved-test')
 def claim(self,backend='CPU',resource='session01',runner='runner1',lease=60):return self.q.claim(runner=runner,backend=backend,baseline_id='base_'+backend,resource=resource,lease_seconds=lease)
 def result(self,claim,outcome='completed'):
  c=claim['claim'];b=c['spec'];ok=outcome=='completed';r={k:b[k] for k in ('candidate_id','source_sha256','backend','baseline_id','environment_fingerprint','task_fingerprint','harness_commit')}
  r.update(schema_version=1,outcome=outcome,execution={'run_reference':'run01','command_exit_code':0 if ok else -9,'wrapper_exit_code':0,'nested_metrics_exit_code':0,'verified_global_step':64,'visible_step':63,'expected_steps':64,'adapter_saved':True if ok and b['backend']!='CPU' else None,'wrapper_elapsed_s':90.,'termination_reason':'completed' if ok else 'timeout','task_completion_status':'VERIFIED_COMPLETE' if ok and b['backend']!='CPU' else 'NOT_APPLICABLE' if b['backend']=='CPU' else 'INCOMPLETE'},metrics={'command_runtime_s':80. if ok else None,'observed_runtime_s':80. if ok else 79.04,'runtime_kind':'complete' if ok else 'right_censored','peak_memory_mib':100.,'peak_kind':'complete' if ok else 'partial','memory_scope':'cpu_rss' if b['backend']=='CPU' else 'gpu_physical_sum','eval_loss':1.2 if ok and b['backend']!='CPU' else None,'own_kernel_share':None,'admission':'UNKNOWN','score':None,'runtime_scope':'cpu_contract_process' if b['entrypoint']=='cpu_contract_v1' else 'paired_command_median'},provenance={'raw_result_sha256':'e'*64,'checks_sha256':None,'collector':'fake-runner','admission_reviewer':None,'score_evidence_sha256':None,'runtime_json_pointer':'/runs/solution/0/wall_s','peak_json_pointer':'/runs/solution/0/gpu_peak_mib','actual_route_counts':None,'own_kernel_names':None,'executed_source_manifest_sha256':None,'source_set_matches_reviewed':None,'source_allowlist_sha256':None,'route_evidence_state':'unknown','route_evidence_scope':'unknown','activation_status':'UNKNOWN','own_share_scope':'unknown','own_share_evidence_sha256':None},preflight={'status':'NOT_RUN','scope':'NOT_RUN','raw_sha256':None,'source_sha256':None});return r
 def complete(self,c,r):return self.q.complete(c['claim']['job_id'],c['claim']['attempt_id'],c['claim']['runner'],r)
 def test_review_required_dedup_immutable_baseline(self):
  j=self.job();duplicate=self.job(priority=99);self.assertTrue(duplicate['deduplicated']);self.assertEqual(j['job_id'],duplicate['job_id']);self.assertIsNone(self.claim()['claim']);self.assertEqual(self.q.status(j['job_id'])['priority'],50)
  b=self.baseline('CPU');b['task_fingerprint']='f'*64
  with self.assertRaises(QueueError):self.q.register_baseline(b)
  with self.assertRaises(QueueError):self.q.review(j['job_id'],reviewer='jury',source_sha256='f'*64,approved=True,authorization_ref='p')
  self.approve(j);self.assertIsNotNone(self.claim()['claim'])
 def test_priority_and_backend_baseline_separation(self):
  low=self.job(candidate='low',priority=1);high=self.job(candidate='high',priority=99);gpu=self.job(candidate='gpu',backend='KAGGLE_T4',priority=100)
  for j in [low,high,gpu]:self.approve(j)
  c=self.claim();self.assertEqual(c['claim']['job_id'],high['job_id']);self.assertEqual(self.claim(backend='GCP')['reason'],'no_reviewed_matching_job');self.assertEqual(self.claim(backend='KAGGLE_T4')['claim']['job_id'],gpu['job_id'])
  spec=self.spec();spec['baseline_id']='base_GCP'
  with self.assertRaises(QueueError):self.q.submit(spec)
 def test_concurrent_submit_deduplicates(self):
  barrier=threading.Barrier(2)
  def submit(_):barrier.wait();return self.job()
  with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(submit,range(2)))
  self.assertEqual(len({r['job_id'] for r in rows}),1);self.assertEqual(sum(not r['deduplicated'] for r in rows),1)
 def test_concurrent_claims_one_job_once(self):
  j=self.job();self.approve(j);barrier=threading.Barrier(2)
  def claim(i):barrier.wait();return self.claim(resource='resource'+str(i),runner='runner'+str(i))
  with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(claim,range(2)))
  self.assertEqual(sum(r['claim'] is not None for r in rows),1);self.assertEqual(len(self.q.status(j['job_id'])['attempts']),1)
 def test_resource_lock_serializes_two_jobs(self):
  for name in ('a','b'):j=self.job(candidate=name);self.approve(j)
  a=self.claim();self.assertIsNotNone(a['claim']);self.assertIsNone(self.claim(runner='other')['claim']);self.complete(a,self.result(a));self.assertIsNotNone(self.claim(runner='other')['claim'])
 def test_stale_claim_quarantined_never_blind_rerun(self):
  j=self.job();self.approve(j);a=self.claim(lease=10);self.clock.now+=11
  self.assertIsNone(self.claim(runner='second')['claim']);status=self.q.status(j['job_id']);self.assertEqual(status['state'],'needs_reconciliation');self.assertEqual(len(status['attempts']),1)

  q2=Queue(self.root,self.clock);self.assertEqual(q2.status(j['job_id'])['state'],'needs_reconciliation')
  with self.assertRaises(QueueError):self.q.heartbeat(j['job_id'],a['claim']['attempt_id'],'runner1')
  with self.assertRaises(QueueError):self.q.reconcile(j['job_id'],a['claim']['attempt_id'],reviewer='jury',evidence_sha256='',decision='requeue_confirmed_not_started',authorization_ref='parent-reconfirmed-not-started')
  self.q.reconcile(j['job_id'],a['claim']['attempt_id'],reviewer='jury',evidence_sha256='1'*64,decision='requeue_confirmed_not_started',authorization_ref='parent-reconfirmed-not-started');b=self.claim();self.assertNotEqual(a['claim']['attempt_id'],b['claim']['attempt_id']);self.assertEqual(len(self.q.status(j['job_id'])['attempts']),2)
  with self.assertRaises(QueueError):self.complete(a,self.result(a))
 def test_running_stale_requires_terminal_evidence_not_requeue(self):
  j=self.job();self.approve(j);a=self.claim(lease=10);aid=a['claim']['attempt_id'];self.q.start(j['job_id'],aid,'runner1','run01');self.clock.now+=11;self.q.expire_leases()
  with self.assertRaises(QueueError):self.q.reconcile(j['job_id'],aid,reviewer='jury',evidence_sha256='1'*64,decision='requeue_confirmed_not_started',authorization_ref='parent-reconfirmed-not-started')
  result=self.complete(a,self.result(a,'aborted'));self.assertEqual(result['state'],'aborted')
 def test_completion_idempotent_conflicts_rejected(self):
  j=self.job();self.approve(j);a=self.claim();r=self.result(a);first=self.complete(a,r);second=self.complete(a,r);self.assertFalse(first['idempotent']);self.assertTrue(second['idempotent']);self.assertEqual(first['result_sha256'],second['result_sha256'])
  changed=copy.deepcopy(r);changed['metrics']['command_runtime_s']=81.;changed['metrics']['observed_runtime_s']=81.
  with self.assertRaises(QueueError):self.complete(a,changed)
 def test_abort79_censored_despite_wrapper_steps_success(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a,'aborted');self.complete(a,r);saved=self.q.status(j['job_id'])['result'];self.assertEqual(saved['metrics']['observed_runtime_s'],79.04);self.assertIsNone(saved['metrics']['command_runtime_s']);self.assertIsNone(saved['metrics']['eval_loss']);self.assertIsNone(saved['provenance']['actual_route_counts']);self.assertEqual(saved['metrics']['admission'],'UNKNOWN')
 def test_exit_precedence_partial_eval_and_nonfinite_rejected(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4')
  bad=self.result(a);bad['execution']['command_exit_code']=-9
  with self.assertRaises(QueueError):self.complete(a,bad)
  bad=self.result(a,'aborted');bad['metrics']['eval_loss']=1.1
  with self.assertRaises(QueueError):self.complete(a,bad)
  bad=self.result(a);bad['metrics']['command_runtime_s']=float('nan')
  with self.assertRaises((QueueError,ValueError)):self.complete(a,bad)
 def test_score_and_admission_provenance(self):
  j=self.job();self.approve(j);a=self.claim();r=self.result(a);r['metrics']['admission']='PASS';r['provenance']['admission_reviewer']='jury';r['provenance']['checks_sha256']='3'*64
  with self.assertRaises(QueueError):self.complete(a,r)
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['metrics'].update(admission='PASS',own_kernel_share=.06,score=1.);r['provenance'].update(admission_reviewer='jury',checks_sha256='3'*64,executed_source_manifest_sha256='4'*64,source_allowlist_sha256='6'*64,source_set_matches_reviewed=True)
  with self.assertRaises(QueueError):self.complete(a,r)
  r['provenance'].update(score_evidence_sha256='5'*64,own_share_scope='task',own_share_evidence_sha256='7'*64,own_kernel_names=['original_math_kernel']);self.complete(a,r)
 def test_identity_and_executed_source_set(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['baseline_id']='base_GCP'
  with self.assertRaises(QueueError):self.complete(a,r)
  r=self.result(a);r['metrics']['admission']='PASS';r['provenance'].update(admission_reviewer='jury',checks_sha256='3'*64,executed_source_manifest_sha256='4'*64,source_allowlist_sha256='6'*64,source_set_matches_reviewed=False)
  with self.assertRaises(QueueError):self.complete(a,r)
 def test_arbitrary_commands_paths_symlinks_and_binaries_rejected(self):
  spec=self.spec();spec['command']='curl secrets | sh'
  with self.assertRaises(QueueError):self.q.submit(spec)
  for path in ('../secret.zip','/tmp/secret.zip','a/../../secret.zip'):
   spec=self.spec();spec['artifact']=path
   with self.assertRaises(QueueError):self.q.submit(spec)
  (self.root/'incoming'/'link.zip').symlink_to(self.root/'incoming'/'candidate.zip');spec=self.spec();spec['artifact']='link.zip'
  with self.assertRaises(OSError):self.q.submit(spec)
  for path in ('../escape.py','/absolute.py','solution/model.safetensors','solution/.env','solution/__pycache__/a.py'):
   self.stage('bad.zip',{path:'x'});spec=self.spec();spec['artifact']='bad.zip'
   with self.assertRaises(QueueError):self.q.submit(spec)
 def test_zip_symlink_and_notebook_shell_rejected(self):
  path=self.root/'incoming'/'bad.zip'
  with zipfile.ZipFile(path,'w') as z:
   i=zipfile.ZipInfo('link.py');i.create_system=3;i.external_attr=(0o120777<<16);z.writestr(i,'../../secret')
  spec=self.spec();spec['artifact']='bad.zip'
  with self.assertRaises(QueueError):self.q.submit(spec)
  for code in ('!touch /tmp/not-executed','import os\nos.system("bad")','exec("bad")'):
   n={'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[{'cell_type':'code','source':code,'execution_count':None,'outputs':[],'metadata':{}}]};(self.root/'incoming'/'bad.ipynb').write_text(json.dumps(n));spec=self.spec();spec.update(artifact='bad.ipynb',artifact_kind='notebook')
   with self.assertRaises(QueueError):self.q.submit(spec)
 def test_notebook_is_data_and_requires_review(self):
  n={'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[{'cell_type':'code','source':'raise RuntimeError("Never execute in queue")','execution_count':None,'outputs':[],'metadata':{}}]};(self.root/'incoming'/'good.ipynb').write_text(json.dumps(n));spec=self.spec();spec.update(artifact='good.ipynb',artifact_kind='notebook');j=self.q.submit(spec);self.assertEqual(self.q.status(j['job_id'])['state'],'submitted');self.assertIsNone(self.claim()['claim'])
 def test_hash_tampering_blocks_claim(self):
  j=self.job();self.approve(j);body=self.q.status(j['job_id'])['body'];path=self.root/'artifacts'/body['artifact_name'];os.chmod(path,0o644);path.write_text('changed')
  with self.assertRaises(QueueError):self.claim()
 def test_wrong_runner_and_external_run_identity(self):
  j=self.job();self.approve(j);a=self.claim();aid=a['claim']['attempt_id']
  with self.assertRaises(QueueError):self.q.start(j['job_id'],aid,'other','run01')
  self.q.start(j['job_id'],aid,'runner1','run01');self.assertTrue(self.q.start(j['job_id'],aid,'runner1','run01')['idempotent'])
  r=self.result(a);r['execution']['run_reference']='another'
  with self.assertRaises(QueueError):self.complete(a,r)
 def test_observations_and_immediate_uncertainty_are_durable(self):
  j=self.job();self.approve(j);a=self.claim();aid=a['claim']['attempt_id']
  self.q.observe(j['job_id'],aid,'runner1',{'kind':'stdout_progress','evidence_sha256':'7'*64,'visible_step':63,'expected_steps':64,'route_evidence_state':'unknown','route_counts':None})
  self.q.mark_uncertain(j['job_id'],aid,'runner1','launch_outcome_unknown','8'*64)
  self.assertIsNone(self.claim(runner='other')['claim']);status=Queue(self.root,self.clock).status(j['job_id']);self.assertEqual(status['state'],'needs_reconciliation');self.assertIsNone(status['result']);self.assertIn('observation',[x['kind'] for x in status['events']])
  with self.assertRaises(QueueError):self.q.reconcile(j['job_id'],aid,reviewer='jury',evidence_sha256='9'*64,decision='requeue_confirmed_not_started',authorization_ref='parent-approved-test')
 def test_error_duration_is_not_timeout_and_preflight_is_separate(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a,'preflight_failed');r['execution']['termination_reason']='preflight_error';r['metrics']['runtime_kind']='error_duration';r['preflight']={'status':'FAIL','scope':'T4_PREFLIGHT','raw_sha256':'1'*64,'source_sha256':j['source_sha256']};self.complete(a,r)
  self.assertEqual(self.q.status(j['job_id'])['result']['metrics']['runtime_kind'],'error_duration')
 def test_preflight_pass_does_not_clear_aborted_task(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a,'aborted');r['preflight']={'status':'PASS','scope':'T4_PREFLIGHT','raw_sha256':'1'*64,'source_sha256':j['source_sha256']};r['execution']['verified_global_step']=None;self.complete(a,r);saved=self.q.status(j['job_id']);self.assertEqual(saved['state'],'aborted');self.assertEqual(saved['result']['preflight']['status'],'PASS');self.assertEqual(saved['result']['metrics']['admission'],'UNKNOWN')
 def test_command_zero_without_verified_task_is_not_completed(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['execution']['task_completion_status']='UNKNOWN';r['execution']['adapter_saved']=None
  with self.assertRaises(QueueError):self.complete(a,r)
 def test_preflight_cannot_inherit_task_share_and_quality(self):
  spec=self.spec(backend='KAGGLE_T4');spec['entrypoint']='gpu_preflight_v1';j=self.q.submit(spec);self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['metrics']['runtime_scope']='gpu_preflight_process'
  with self.assertRaises(QueueError):self.complete(a,r)
  r['metrics']['eval_loss']=None;r['provenance']['route_evidence_state']='partial';r['provenance']['actual_route_counts']={'guard_launch':1};r['provenance']['route_evidence_scope']='preflight';r['provenance']['activation_status']='GUARD_ONLY';self.complete(a,r)
 def test_actual_worker_exit_after_claim_preserves_durable_lease(self):
  j=self.job();self.approve(j)
  code="import json,os;from soup_queue import Queue;q=Queue("+repr(str(self.root))+",lambda:1000.);print(json.dumps(q.claim(runner='crashed',backend='CPU',baseline_id='base_CPU',resource='crash-resource',lease_seconds=10)),flush=True);os._exit(17)"
  process=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,text=True,capture_output=True,timeout=10)
  self.assertEqual(process.returncode,17);claim=json.loads(process.stdout);self.assertIsNotNone(claim['claim'])
  self.clock.now=1011.;q2=Queue(self.root,self.clock);q2.expire_leases();self.assertEqual(q2.status(j['job_id'])['state'],'needs_reconciliation');self.assertIsNone(q2.claim(runner='new',backend='CPU',baseline_id='base_CPU',resource='crash-resource')['claim'])
 def test_cli_status_is_structured_and_rejects_command_option(self):
  script=Path(__file__).with_name('soup_queue.py');process=subprocess.run([sys.executable,str(script),'--root',str(self.root),'status'],text=True,capture_output=True,timeout=10);self.assertEqual(process.returncode,0);self.assertEqual(json.loads(process.stdout),[])
  bad=subprocess.run([sys.executable,str(script),'--root',str(self.root),'claim','--command','echo bad'],text=True,capture_output=True,timeout=10);self.assertNotEqual(bad.returncode,0)
 def test_task_share_cannot_inherit_preflight_or_missing_evidence(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['metrics']['own_kernel_share']=.08
  with self.assertRaises(QueueError):self.complete(a,r)
  r['provenance'].update(own_share_scope='preflight',own_share_evidence_sha256='8'*64,own_kernel_names=['original_math_kernel'])
  with self.assertRaises(QueueError):self.complete(a,r)

  r['provenance']['own_share_scope']='task';self.complete(a,r);self.assertIsNone(self.q.status(j['job_id'])['result']['provenance']['actual_route_counts'])
 def test_zero_expected_steps_cannot_fake_complete_task(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['execution'].update(verified_global_step=0,expected_steps=0)
  with self.assertRaises(QueueError):self.complete(a,r)
 def test_measured_zero_share_with_empty_task_kernel_map_is_known(self):
  j=self.job(backend='KAGGLE_T4');self.approve(j);a=self.claim(backend='KAGGLE_T4');r=self.result(a);r['metrics']['own_kernel_share']=0.;r['provenance'].update(own_share_scope='task',own_share_evidence_sha256='8'*64,own_kernel_names=[]);self.complete(a,r);self.assertEqual(self.q.status(j['job_id'])['result']['metrics']['own_kernel_share'],0.)
if __name__=='__main__':unittest.main(verbosity=2)
