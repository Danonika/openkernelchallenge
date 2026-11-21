"""Bounded CPU tests. GPU-shaped records are mocks, never actual GPU results."""
import hashlib,io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from soup_queue import Queue,QueueError,digest
import runner_adapter as R

class LocalClient:
    def __init__(self,q):self.q=q;self.calls=[]
    def request(self,method,path,body=None):
        self.calls.append((method,path,body))
        if path=='/v1/claims':return self.q.claim(runner='api-worker',**body)
        parts=path.split('/');jid,aid,action=parts[3],parts[5],parts[6]
        if action=='start':return self.q.start(jid,aid,'api-worker',body['run_reference'])
        if action=='heartbeat':return self.q.heartbeat(jid,aid,'api-worker',body['lease_seconds'])
        if action=='complete':return self.q.complete(jid,aid,'api-worker',body)
        if action=='uncertain':return self.q.mark_uncertain(jid,aid,'api-worker',body['reason'],body['evidence_sha256'])
        raise AssertionError(path)
    def download(self,jid,destination,expected_sha256):
        job=self.q.status(jid);raw=self.q._blob(job['body']).read_bytes();assert digest(raw)==expected_sha256
        Path(destination).write_bytes(raw)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.q=Queue(self.root/'queue');self.client=LocalClient(self.q)
        self.baseline={'baseline_id':'cpu-smoke-only','backend':'CPU','environment_fingerprint':'1'*64,'task_fingerprint':'2'*64,'harness_commit':R.COMMIT,'evidence_sha256':'3'*64};self.baseline_hash=self.q.register_baseline(self.baseline)['hash']
        self.config={'queue_url':'http://127.0.0.1:8765','workspace_root':str(self.root/'work'),'python':sys.executable,'backend':'CPU','baseline_id':'cpu-smoke-only','resource':'cpu-lane','environment_fingerprint':'1'*64,'task_fingerprint':'2'*64,'harness_commit':R.COMMIT,'baseline_fingerprint':self.baseline_hash,'trusted_cpu_smoke_ack':True,'lease_seconds':600,'cpu_affinity':sorted(os.sched_getaffinity(0))[:2],'profiles':{}}
    def tearDown(self):self.temp.cleanup()
    def submit(self,source,timeout=5,extra=None):
        files={'solution/__init__.py':source.encode()};files.update(extra or {});buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            for n,b in files.items():z.writestr(n,b)
        raw=buf.getvalue();p=self.q.root/'incoming/source.zip';p.write_bytes(raw)
        spec={'candidate_id':'reviewed-cpu-smoke','backend':'CPU','baseline_id':self.baseline['baseline_id'],'entrypoint':'cpu_contract_v1','artifact':'source.zip','artifact_kind':'solution_bundle','priority':1,'submitted_by':'test','contains_secrets':False}
        job=self.q.submit(spec);h=job['source_sha256'];self.q.review(job['job_id'],reviewer='reviewer',source_sha256=h,approved=True,authorization_ref='test-cpu-only')
        self.config['profiles'][h]={'kind':'cpu_smoke_v1','files':{n:digest(b) for n,b in files.items()},'module':'solution','timeout_s':timeout}
        return job
    def test_real_child_smoke_and_no_fabricated_admission(self):
        self.submit('import os\ndef queue_smoke():\n return {"passed":sum(i*i for i in range(5))==30,"pid":os.getpid(),"admission":"PASS","score":999}\n')
        out=R.consume_once(self.client,self.config);r=out['result'];raw=json.loads((Path(out['run_dir'])/'runner_raw.json').read_text())
        self.assertEqual(r['outcome'],'completed');self.assertNotEqual(raw['child_result']['candidate_report']['pid'],os.getpid());self.assertEqual(r['metrics']['admission'],'UNKNOWN');self.assertIsNone(r['metrics']['score']);self.assertEqual(r['metrics']['runtime_scope'],'cpu_contract_process');self.assertIsNone(r['metrics']['eval_loss'])
    def test_environment_secrets_are_not_inherited(self):
        self.submit('import os\ndef queue_smoke():\n return {"passed":not any(k in os.environ for k in ("SOUP_QUEUE_WORKER_TOKEN","HF_TOKEN","GOOGLE_APPLICATION_CREDENTIALS","AWS_SECRET_ACCESS_KEY","PYTHONPATH")),"home":os.environ["HOME"]}\n')
        with patch.dict(os.environ,{'SOUP_QUEUE_WORKER_TOKEN':'secret','HF_TOKEN':'secret','GOOGLE_APPLICATION_CREDENTIALS':'secret','AWS_SECRET_ACCESS_KEY':'secret','PYTHONPATH':'secret'}):out=R.consume_once(self.client,self.config)
        self.assertEqual(out['result']['outcome'],'completed')
    def test_actual_nonzero_child_fails_and_no_eval(self):
        self.submit('def queue_smoke():\n raise ValueError("intentional CPU fixture failure")\n')
        r=R.consume_once(self.client,self.config)['result'];self.assertEqual(r['outcome'],'failed');self.assertEqual(r['execution']['command_exit_code'],1);self.assertEqual(r['metrics']['runtime_kind'],'error_duration');self.assertIsNone(r['metrics']['eval_loss'])
    def test_timeout_is_censored(self):
        self.submit('import time\ndef queue_smoke():\n time.sleep(20)\n return {"passed":True}\n',timeout=1)
        r=R.consume_once(self.client,self.config)['result'];self.assertEqual(r['outcome'],'aborted');self.assertEqual(r['execution']['termination_reason'],'timeout');self.assertEqual(r['metrics']['runtime_kind'],'right_censored');self.assertIsNone(r['metrics']['command_runtime_s'])
    def test_unapproved_local_hash_does_not_execute(self):
        job=self.submit('raise RuntimeError("must not import")\n');self.config['profiles']={}
        with self.assertRaises(QueueError):R.consume_once(self.client,self.config)
        self.assertEqual(self.q.status(job['job_id'])['state'],'needs_reconciliation');self.assertFalse(any(c[1].endswith('/start') for c in self.client.calls))
    def test_exact_source_allowlist_rejects_extra(self):
        job=self.submit('def queue_smoke(): return {"passed":True}\n',extra={'extra.py':b'pass\n'});self.config['profiles'][job['source_sha256']]['files'].pop('extra.py')
        with self.assertRaises(QueueError):R.consume_once(self.client,self.config)
        self.assertFalse(list((self.root/'work').rglob('child.log')))
    def test_no_work_is_not_fake_success(self):
        r=R.consume_once(self.client,self.config);self.assertIsNone(r['claim']);self.assertEqual(r['reason'],'no_reviewed_matching_job')
    def test_corrupt_source_hash_refused(self):
        self.submit('def queue_smoke(): return {"passed":True}\n')
        old=self.client.download
        def corrupt(*args):old(*args);Path(args[1]).write_bytes(b'bad')
        self.client.download=corrupt
        with self.assertRaises(QueueError):R.consume_once(self.client,self.config)
    def mock_screen(self,exit=0,step=64,wall=100,adapter=True,wrapper=0,timeout=False):
        spec={'candidate_id':'mock-only','source_sha256':'a'*64,'backend':'GCP','baseline_id':'gcp-host-local','environment_fingerprint':'b'*64,'task_fingerprint':'c'*64,'harness_commit':R.COMMIT,'entrypoint':'soup_screen_v1'}
        process={'exit_code':wrapper,'elapsed_s':110,'timed_out':False,'heartbeat_error':None,'surviving_owned_pids':[]}
        native={'exit_code':exit,'wall_s':wall,'gpu_peak_mib':14345,'global_step':step,'adapter_tensors':{'x':['F16',[1]]} if adapter else None,'adapter_config':{'r':16},'metrics':{'exit_code':0,'own_kernel_ms':{'bad_claim':99999}}}
        record={'process':process,'harness_result':{'runs':{'solution':[native]},'evals':{'solution':{'exit_code':0,'mean_loss':2.0}},'packages_changed':[],'admission':'PASS','score':999},'adapter_saved_on_disk':adapter,'native_timeout':timeout,'visible_step':step}
        return spec,record
    def test_mock_gpu_normalizer_no_admission_or_share_inference(self):
        spec,record=self.mock_screen();r=R.normalize_screen(spec,'run_test',record,'d'*64,'e'*64);R.validate_result(r,spec)
        self.assertEqual(r['outcome'],'completed');self.assertEqual(r['metrics']['command_runtime_s'],100);self.assertEqual(r['metrics']['eval_loss'],2);self.assertIsNone(r['metrics']['own_kernel_share']);self.assertEqual(r['metrics']['admission'],'UNKNOWN');self.assertIsNone(r['metrics']['score'])
    def test_mock_native_timeout_overrides_nested_zero(self):
        spec,record=self.mock_screen(exit=-9,step=64,wall=79.04,timeout=True);r=R.normalize_screen(spec,'run_test',record,'d'*64,'e'*64);R.validate_result(r,spec)
        self.assertEqual(r['outcome'],'aborted');self.assertEqual(r['metrics']['peak_kind'],'partial');self.assertIsNone(r['metrics']['eval_loss']);self.assertIsNone(r['provenance']['actual_route_counts'])
    def test_mock_incomplete_task_not_success(self):
        spec,record=self.mock_screen(step=63,adapter=False);r=R.normalize_screen(spec,'run_test',record,'d'*64,'e'*64);R.validate_result(r,spec);self.assertEqual(r['outcome'],'failed');self.assertIsNone(r['metrics']['command_runtime_s'])
    def test_mock_missing_result_no_native_runtime_invention(self):
        spec,record=self.mock_screen();record['harness_result']=None;record['process'].update(exit_code=-9,timed_out=True)
        r=R.normalize_screen(spec,'run_test',record,'d'*64,'e'*64);R.validate_result(r,spec);self.assertEqual(r['metrics']['runtime_kind'],'unavailable');self.assertEqual(r['execution']['wrapper_elapsed_s'],110);self.assertIsNone(r['metrics']['observed_runtime_s'])
    def test_owned_detached_descendant_cleanup(self):

        run=self.root/'owned';run.mkdir();env=R.child_env(run,self.config)
        code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); print(p.pid,flush=True); time.sleep(30)'
        result=R.supervised_process([sys.executable,'-c',code],str(run),env,0.6,run/'log')
        self.assertTrue(result['timed_out']);self.assertGreaterEqual(len(result['owned_descendants_killed']),2);self.assertEqual(result['surviving_owned_pids'],[])

    def docker_config(self):
        c=dict(self.config);c.update(backend='GCP',execution_boundary='docker_v1',docker_binary=sys.executable,container_image='registry.example/soup@sha256:'+'4'*64,hf_hub_cache=str(self.root/'public-cache'),container_uid=10001 if os.getuid()!=10001 else 10002,container_gid=10001,cpu_affinity=sorted(os.sched_getaffinity(0))[:4]);return c
    def test_docker_argv_no_credentials_no_network_no_arbitrary_command(self):
        c=self.docker_config();run=self.root/'isolated';run.mkdir()
        argv=R.container_argv(c,run,'soupq-test','harness',timeout_s=321)
        self.assertEqual(argv[argv.index('--network')+1],'none');self.assertIn('--read-only',argv);self.assertEqual(argv[argv.index('--cap-drop')+1],'ALL');self.assertIn('no-new-privileges:true',argv);self.assertIn('--pull=never',argv);self.assertEqual(argv[argv.index('--gpus')+1],'device=0')
        mounts=[argv[i+1] for i,v in enumerate(argv) if v=='--mount'];self.assertEqual(len(mounts),4);self.assertEqual(sum('readonly' in v for v in mounts),3)
        self.assertFalse(any('docker.sock' in v or 'queue.sqlite' in v or 'WORKER_TOKEN' in v for v in argv));self.assertNotIn('--privileged',argv);self.assertNotIn('--pid',argv)
        self.assertEqual(argv[-4:],['--timeout','321','--out','/output/runs'])
        with self.assertRaises(QueueError):R.container_argv(c,run,'soupq-test','submitted-shell')
    def test_missing_docker_fails_closed_before_claim(self):
        c=self.docker_config();c['docker_binary']=str(self.root/'missing-docker');c['profiles']={}
        with self.assertRaises(QueueError):R.consume_once(self.client,c)
        self.assertEqual(self.client.calls,[])
    def test_disabled_gpu_fails_before_claim(self):
        c=self.docker_config();c['profiles']={};c['gpu_execution_enabled']=False
        with self.assertRaises(QueueError):R.consume_once(self.client,c)
        self.assertEqual(self.client.calls,[])
    def test_container_policy_rejects_host_or_secret_mount(self):
        c=self.docker_config();inspection={'HostConfig':{'NetworkMode':'none','ReadonlyRootfs':True,'Privileged':False,'PidMode':'','IpcMode':'private','CapDrop':['ALL'],'SecurityOpt':['no-new-privileges:true']},'Config':{'User':f"{c['container_uid']}:{c['container_gid']}"},'Mounts':[{'Destination':p,'RW':rw} for p,rw in [('/task',False),('/hf_cache',False),('/runner',False),('/output',True)]]}
        self.assertTrue(R.check_container_policy(inspection,c));inspection['HostConfig']['NetworkMode']='host';self.assertFalse(R.check_container_policy(inspection,c));inspection['HostConfig']['NetworkMode']='none';inspection['Mounts'].append({'Destination':'/secrets','RW':False});self.assertFalse(R.check_container_policy(inspection,c))
    def test_fixed_host_lock_serializes_baseline_and_candidate(self):
        with patch.object(R,'GPU_LOCK_PATH',self.root/'host-gpu.lock'):
            with R.host_gpu_lock():
                with self.assertRaises(QueueError):
                    with R.host_gpu_lock():pass
            with R.host_gpu_lock():pass
    def test_unknown_cleanup_quarantine_blocks_host_lane(self):
        p=self.root/'host-gpu.lock'
        Path(str(p)+'.blocked').write_text('{}')
        with patch.object(R,'GPU_LOCK_PATH',p):
            with self.assertRaises(QueueError):
                with R.host_gpu_lock():pass
    def test_baseline_requires_finite_evaluation(self):
        self.assertTrue(R.finite_eval({'exit_code':0,'mean_loss':2.0}))
        for value in (None,float('nan'),float('inf'),-1):self.assertFalse(R.finite_eval({'exit_code':0,'mean_loss':value}))
    def test_cpu_interruption_has_unknown_contract_status(self):
        self.submit('import time\ndef queue_smoke():\n time.sleep(20)\n return {"passed":True}\n',timeout=1)
        r=R.consume_once(self.client,self.config)['result'];self.assertEqual(r['preflight']['status'],'UNKNOWN')

    def test_container_outputs_refuse_symlinks_fifo_and_oversize(self):
        output=self.root/'container_output';output.mkdir();secret=self.root/'fake_private.json';secret.write_text('{"synthetic":"not a real secret"}')
        (output/'result.json').symlink_to(secret)
        with self.assertRaises((OSError,QueueError)):R.output_bytes(output,'result.json')
        (output/'nested').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises((OSError,QueueError)):R.output_bytes(output,'nested/fake_private.json')
        os.mkfifo(output/'train.log')
        with self.assertRaises(QueueError):R.output_bytes(output,'train.log')
        (output/'large').write_bytes(b'12345')
        with self.assertRaises(QueueError):R.output_bytes(output,'large',limit=4)
        self.assertEqual(R.output_bytes(output,'large',limit=5),b'12345')
        self.assertIsNone(R.output_bytes(output,'missing',optional=True))
        with self.assertRaises((OSError,QueueError)):R.output_present(output,'result.json')
        self.assertTrue(R.output_present(output,'large'))
    @unittest.skipIf(os.geteuid()==0,'Read-permission regression requires a nonroot test user')
    def test_private_adapter_metadata_probe(self):
        output=self.root/'private_output';output.mkdir()
        weights=output/'adapter.safetensors';weights.write_bytes(b'private adapter bytes');weights.chmod(0)
        try:
            self.assertFalse(os.access(weights,os.R_OK))
            self.assertTrue(R.output_present(output,'adapter.safetensors'))
            with self.assertRaises(PermissionError):R.output_bytes(output,'adapter.safetensors')
        finally:weights.chmod(0o600)
        (output/'empty').touch()
        self.assertFalse(R.output_present(output,'empty'))
        self.assertFalse(R.output_present(output,'absent'))
        os.mkfifo(output/'fifo')
        with self.assertRaises(QueueError):R.output_present(output,'fifo')
    def test_immutable_local_image_id_and_boundary_config_binding(self):
        c=self.docker_config();c['container_image']='sha256:'+'4'*64
        self.assertEqual(R.docker_spec(c)[1],c['container_image'])
        binding=R.boundary_binding(c);c['container_gid']+=1;self.assertNotEqual(binding,R.boundary_binding(c))
        self.assertIn('runner_adapter.py',R.boundary_helpers())
        c['container_image']='soup-image:latest'
        with self.assertRaises(QueueError):R.docker_spec(c)
    def test_candidate_baseline_timeout_budget_mismatch_refused(self):
        c=self.docker_config();c.update(baseline_timeout_s=900,baseline_outer_timeout_s=4000)
        c['profiles']={'a'*64:{'kind':'soup_screen_v1','files':{'solution/__init__.py':'b'*64},'timeout_s':79,'outer_timeout_s':4000,'expected_steps':64}}
        with self.assertRaises(QueueError):R.checked_config(c)
        c['profiles']['a'*64]['timeout_s']=900;self.assertEqual(R.checked_config(c)['baseline_timeout_s'],900)

    def test_official_data_and_model_revision_cannot_be_reconfigured(self):
        wrong=dict(R.PUBLIC_DATA);wrong['data/train.jsonl']='a'*64
        with self.assertRaisesRegex(QueueError,'Exact pinned public'):
            R.stage_official({'data_files':wrong},self.root/'unused')
        manifest=self.root/'wrong-model.json';manifest.write_text(json.dumps({'model_id':R.MODEL,'revision':'a'*40,'files':{}}))
        with self.assertRaisesRegex(QueueError,'fixed model/revision'):
            R.verify_cache({'model_cache_manifest':str(manifest),'model_cache_manifest_sha256':R.file_hash(manifest)})

if __name__=='__main__':unittest.main(verbosity=2)
