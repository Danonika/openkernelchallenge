"""Bounded native/candidate original NF4 forward MICROBENCH_ONLY orchestration.

Never installs into SOUP_REPO or calls a training harness. Existing source/runtime,
lock, pidfd/subreaper, quarantine and idle-check helpers are preserved unchanged.
The deployed service's limits remain authoritative; each GPU child gets <=80 s.
"""
import fcntl, hashlib, json, os, sys, time, uuid
from pathlib import Path
import process_guard
import owned_process_tree as tree
from reviewed_support import verify_official, idle_check, Exporter, export_small, read, write, sha

LOCK = Path('/tmp/soup-gpu0.lock')
MARKER = Path('/tmp/soup-gpu0.lock.blocked')

def main():
    here = Path(__file__).resolve().parent
    if not os.environ.get('SOUP_REPO') or not os.environ.get('SOUP_RESULTS'):
        raise SystemExit('Requires deployed SOUP_REPO/SOUP_RESULTS, no manual fallback')
    repo, results = Path(os.environ['SOUP_REPO']).absolute(), Path(os.environ['SOUP_RESULTS']).absolute()
    if repo.is_symlink() or results.is_symlink() or repo == results:
        raise SystemExit('Prepared checkout/results must be separate real directories')
    assert Path('/kaggle/working').is_dir()
    results.mkdir(parents=True,exist_ok=True)
    identity = 'nf4-mma-microbench-only-' + uuid.uuid4().hex[:12]
    run, export = Path('/kaggle/working')/identity, results/identity
    run.mkdir(exist_ok=False); export.mkdir(exist_ok=False)
    status = dict(status='PREPARING_MICROBENCH_ONLY',scope='MICROBENCH_ONLY',
                  training_executed=False,complete_fixed_task=False,score=None,admission='UNKNOWN',
                  per_gpu_child_wall_cap_s=80,cap_bypass=False,
                  candidate_source_sha256=read(here/'source_manifest.json'),
                  methods_in_execution_order=['native','candidate'],processes={})
    write(export/'status.json',status)
    lockfd = None; started = time.monotonic()
    try:
        for name,digest in read(here/'payload_manifest.json').items():
            p = here/name
            assert not p.is_symlink() and p.is_file() and sha(p)==digest, 'Payload changed: '+name
        if MARKER.exists(): raise RuntimeError('GPU quarantine requires operator reconciliation')
        lockfd = os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if MARKER.exists(): raise RuntimeError('GPU quarantine appeared while locking')
        tree.prepare(); tree.begin_quarantine(MARKER)
        official = read(here/'official_manifest.json')
        verify_official(repo,official)
        env = dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env.pop('SOUP_ROTARY_CODEGEN_DIR',None)
        with Exporter(run,export):
            probe = process_guard.run([sys.executable,'-B',str(here/'runtime_probe.py'),str(run/'runtime_probe.json')],
                                      cwd=repo,env=env,log=run/'runtime_probe.log',timeout=40,lock_fd=lockfd)
            status['processes']['runtime_probe'] = probe
            if probe['exit_code'] or not read(run/'runtime_probe.json').get('passed'):
                raise RuntimeError('Exact deployed runtime gate failed; no benchmark')
            def idle(label):
                folder=run/('idle_'+label);folder.mkdir(exist_ok=False)
                result=idle_check(env,repo,folder,lockfd)
                status['processes']['idle_'+label]=result
                if result['exit_code']:raise RuntimeError('Idle gate failed: '+label)
            idle('before_preflight')
            # Existing original numerical/PTX gate, isolated from every measured cache.
            preenv = dict(env,TRITON_CACHE_DIR=str(run/'preflight_triton'),
                          TORCHINDUCTOR_CACHE_DIR=str(run/'preflight_inductor'))
            pre = process_guard.run([sys.executable,'-B','-m','solution.web_preflight','--out',str(run/'preflight.json')],
                                    cwd=here,env=preenv,log=run/'preflight.log',timeout=80,lock_fd=lockfd)
            status['processes']['preflight'] = pre
            if pre['exit_code'] or read(run/'preflight.json').get('status')!='PASSED_PREFLIGHT_ONLY':
                raise RuntimeError('Original NF4 MMA numerical/PTX gate failed; no benchmark')
            status['status']='RUNNING_MICROBENCH_ONLY';write(export/'status.json',status)
            idle('preflight')
            for method in status['methods_in_execution_order']:
                cache=run/'triton_cache'/method;cache.mkdir(parents=True,exist_ok=False)
                childenv=dict(env,TRITON_CACHE_DIR=str(cache),TORCHINDUCTOR_CACHE_DIR=str(run/'inductor_cache'/method))
                process=process_guard.run([sys.executable,'-B',str(here/'worker.py'),'--method',method,'--out',str(run/(method+'.json'))],
                                          cwd=here,env=childenv,log=run/(method+'.log'),timeout=80,lock_fd=lockfd)
                status['processes'][method]=process
                if process['exit_code'] or read(run/(method+'.json')).get('status')!='PASSED_MICROBENCH_ONLY':
                    raise RuntimeError('Incomplete/failed/censored microbenchmark: '+method)
                idle(method)
                write(export/'status.json',status)
            verify_official(repo,official)
            reports={method:read(run/(method+'.json')) for method in status['methods_in_execution_order']}
            assert all(len(r['rows'])==6 and r['training_executed'] is False and r['complete_fixed_task'] is False for r in reports.values())
            write(run/'comparison.json',dict(scope='MICROBENCH_ONLY',reports=reports,
                                            score=None,admission='UNKNOWN',complete_fixed_task=False,
                                            note='Raw first-call wall and warm CUDA-event samples, not pure compile cost or task speedup; no ratios or ownership-share inference.'))
            status['status']='PASSED_MICROBENCH_ONLY'
    except BaseException as exc:
        status.update(status='BLOCKED_FAILED_OR_CENSORED_MICROBENCH_ONLY',error=repr(exc))
        if lockfd is not None:tree.quarantine('NF4 microbenchmark failed: '+repr(exc))
    finally:
        status['orchestrator_wall_s']=time.monotonic()-started
        export_small(run,export);write(export/'status.json',status)
        if lockfd is not None:
            tree.finish_quarantine();os.close(lockfd)
    print(json.dumps({'scope':'MICROBENCH_ONLY','status':status['status'],'export':str(export),
                      'score':None,'admission':'UNKNOWN'}),flush=True)
    raise SystemExit(0 if status['status']=='PASSED_MICROBENCH_ONLY' else 1)

if __name__=='__main__':main()
