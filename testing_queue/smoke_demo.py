#!/usr/bin/env python3
"""One real local submit-review-claim-child-result smoke. Dummy credentials, no GPU/network install."""
import argparse,base64,hashlib,json,os,platform,sys,threading
from pathlib import Path
from api import LocalServer
from agent_client import Client
from bundle_candidate import bundle
from runner_adapter import consume_once,COMMIT
DUMMY={role:'TEST_ONLY_SMOKE_'+role.upper()+'_'+'x'*32 for role in ('submitter','reviewer','worker')}
def run(out):
 out=Path(out).absolute();out.mkdir(mode=0o700,parents=False)
 server=LocalServer(out/'queue',DUMMY,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  url='http://127.0.0.1:'+str(server.server_port);reviewer=Client(url,DUMMY['reviewer']);submitter=Client(url,DUMMY['submitter']);worker=Client(url,DUMMY['worker'])
  raw,files=bundle(Path(__file__).resolve().parent/'smoke/solution');source=hashlib.sha256(raw).hexdigest();(out/'smoke.zip').write_bytes(raw)
  environment=hashlib.sha256(json.dumps({'python':platform.python_version(),'platform':platform.platform()},sort_keys=True).encode()).hexdigest();task=hashlib.sha256(b'infrastructure-smoke-sum-of-squares-v1-not-training').hexdigest()
  baseline={'baseline_id':'cpu-smoke-local-v1','backend':'CPU','environment_fingerprint':environment,'task_fingerprint':task,'harness_commit':COMMIT,'evidence_sha256':task};registered=reviewer.request('POST','/v1/baselines',baseline)
  receipt=submitter.request('POST','/v1/jobs',{'candidate_id':'QUEUE-SELFTEST-ONLY','backend':'CPU','baseline_id':baseline['baseline_id'],'entrypoint':'cpu_contract_v1','artifact_kind':'solution_bundle','priority':10,'contains_secrets':False,'artifact_base64':base64.b64encode(raw).decode()})
  reviewer.request('POST','/v1/jobs/'+receipt['job_id']+'/review',{'source_sha256':source,'approved':True,'authorization_ref':'explicit-bundled-infrastructure-selftest'})
  config={'queue_url':url,'workspace_root':str(out/'worker'),'python':sys.executable,'backend':'CPU','baseline_id':baseline['baseline_id'],'baseline_fingerprint':registered['hash'],'resource':'selftest-cpu','environment_fingerprint':environment,'task_fingerprint':task,'harness_commit':COMMIT,'lease_seconds':600,'cpu_affinity':sorted(os.sched_getaffinity(0))[:2],'execution_enabled':True,'trusted_cpu_smoke_ack':True,'profiles':{source:{'kind':'cpu_smoke_v1','files':files,'module':'solution','timeout_s':10}}}
  result=consume_once(worker,config);job=submitter.request('GET','/v1/jobs/'+receipt['job_id'])
  assert job['state']=='completed' and job['result']['metrics']['score'] is None and job['result']['metrics']['admission']=='UNKNOWN'
  summary={'passed':True,'scope':'CPU infrastructure child execution only; not training/GPU/performance','job_id':receipt['job_id'],'source_sha256':source,'child_executed':True,'result':job['result'],'worker_run_dir':result['run_dir'],'remote_actions':0,'real_credentials_used':False}
  (out/'smoke_result.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
 finally:server.shutdown();server.server_close();thread.join(timeout=5)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out),indent=2))
