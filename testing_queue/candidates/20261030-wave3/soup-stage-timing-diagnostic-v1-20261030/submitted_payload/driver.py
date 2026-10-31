"""One fixed64-step native solution-side diagnostic, not a scored paired run.

Unchanged official harness measures the entire command under native80s and the
service's authoritative hard cap. No caches, outputs or task work are bypassed.
"""
import fcntl, json, os, sys, time, uuid
from pathlib import Path
import process_guard
import owned_process_tree as tree
import reviewed_support as support
from evidence_reader import events

MODE='stage_timing_diagnostic'
LOCK=Path('/tmp/soup-gpu0.lock')
MARKER=Path('/tmp/soup-gpu0.lock.blocked')


def diagnostic_errors(raw,process,log):
    errors=[]
    if process.get('exit_code')!=0 or process.get('timed_out') or process.get('interrupted') or process.get('cleanup_confirmed') is not True:errors.append('harness_process_incomplete')
    task=raw.get('task',{})
    if task.get('only')!='solution' or task.get('repeats')!=1 or task.get('seed')!=1:errors.append('single_solution_scope')
    if raw.get('packages_changed')!=[]:errors.append('package_inventory')
    if set(raw.get('runs',{}))!={'solution'}:errors.append('single_side')
    rows=raw.get('runs',{}).get('solution',[])
    if len(rows)!=1:return errors+['single_run_missing']
    row=rows[0];m=row.get('metrics') or {}
    if row.get('run')!='solution_1' or row.get('side')!='solution' or row.get('exit_code')!=0 or m.get('exit_code')!=0 or m.get('steps')!=64 or row.get('global_step')!=64:errors.append('native64_command_incomplete')
    errors+=support.output_problems(row)
    if not support.finite(row.get('wall_s')) or row['wall_s']<=0 or not support.finite(row.get('gpu_peak_mib')) or row['gpu_peak_mib']<=0:errors.append('native_measurement_missing')
    if m.get('own_kernel_ms')!={} or m.get('solution') is not True:errors.append('not_native_no_kernel_control')
    ev=raw.get('evals',{}).get('solution',{})
    if ev.get('exit_code')!=0 or ev.get('rows')!=256 or not support.finite(ev.get('mean_loss')) or not isinstance(ev.get('tokens'),int) or ev['tokens']<=0:errors.append('native_public_evaluation_missing')
    records=events(log,'SOUP_STAGE_TIMING ')
    for label in ('apply','soup_setup','model_setup','trainer_init','trainer_train','adapter_save','soup_train_and_final_save'):
        if not any(x.get('stage')==label and x.get('event')=='return' for x in records):errors.append('stage_return_missing:'+label)
    if 'TIMEOUT' in log or 'TIME_LIMIT_EXCEEDED' in log:errors.append('native_deadline')
    return errors


def main():
    here=Path(__file__).resolve().parent
    if not os.environ.get('SOUP_REPO') or not os.environ.get('SOUP_RESULTS'):raise SystemExit('Deployed paths required; no fallback')
    repo=Path(os.environ['SOUP_REPO']).absolute();results=Path(os.environ['SOUP_RESULTS']).absolute()
    if repo.is_symlink() or results.is_symlink() or repo==results:raise SystemExit('Distinct real job paths required')
    if not Path('/kaggle/working').is_dir():raise SystemExit('Deployed working directory missing')
    results.mkdir(parents=True,exist_ok=True)
    identity='stage-timing-only-'+uuid.uuid4().hex[:12];run=Path('/kaggle/working')/identity;export=results/identity
    run.mkdir(exist_ok=False);export.mkdir(exist_ok=False)
    status={'status':'PREPARING_STAGE_DIAGNOSTIC_ONLY','scope':'SINGLE_FIXED_TASK_STAGE_DIAGNOSTIC_ONLY','mode':MODE,'source_commit_reference':support.COMMIT,'native_timeout_requested_s':80,'cap_bypass':False,'paired':False,'complete_fixed_task':False,'complete_native_command_and_eval':False,'score':None,'admission':'UNKNOWN','official_kaggle_admission':'NOT_ESTABLISHED'}
    support.write(export/'status.json',status);lockfd=None;started=time.monotonic()
    try:
        for name,h in support.read(here/'payload_manifest.json').items():
            path=here/name
            if path.is_symlink() or not path.is_file() or support.sha(path)!=h:raise RuntimeError('Payload changed: '+name)
        if MARKER.exists():raise RuntimeError('GPU quarantine requires operator reconciliation')
        lockfd=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if MARKER.exists():raise RuntimeError('GPU quarantine appeared while locking')
        tree.prepare();tree.begin_quarantine(MARKER)
        official=support.read(here/'official_manifest.json');support.verify_official(repo,official)
        env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env.pop('SOUP_ROTARY_CODEGEN_DIR',None)
        with support.Exporter(run,export):
            probe=process_guard.run([sys.executable,'-B',str(here/'runtime_probe.py'),str(run/'runtime_probe.json')],cwd=repo,env=env,log=run/'runtime_probe.log',timeout=40,lock_fd=lockfd)
            runtime=support.read(run/'runtime_probe.json')
            if probe['exit_code'] or not runtime.get('passed'):raise RuntimeError('Exact runtime gate failed')
            status['runtime']=runtime
            from prepare_asset_manifest import build
            assets,data=build(runtime['resolved_public_hub_cache'],repo);support.write(run/'model_cache_manifest.json',assets);support.write(run/'public_data_manifest.json',data)
            support.FUSIONS=support.read(here/'candidate_registry.json')
            bundle=here/(MODE+'.zip');manifest=support.read(here/'source_manifest.json')
            support.install_bundle(repo,bundle,manifest,run/'original_solution');support.write(run/'candidate_source_manifest.json',manifest);status['source_sha256']=support.sha(bundle)
            pre=process_guard.run([sys.executable,'-B','-m','solution.preflight','--out',str(run/'preflight.json')],cwd=repo,env=env,log=run/'preflight.log',timeout=20,lock_fd=lockfd)
            if pre['exit_code'] or support.read(run/'preflight.json').get('status')!='PASS_CPU_OBSERVATIONAL_WRAPPERS_ONLY':raise RuntimeError('CPU wrapper preflight failed')
            idle_dir=run/'idle_preflight';idle_dir.mkdir();idle=support.idle_check(env,repo,idle_dir,lockfd)
            if idle['exit_code']:raise RuntimeError('GPU idle gate failed')
            support.verify_official(repo,official);out=run/'official_output'
            command=[sys.executable,'-B',str(repo/'harness/run_task.py'),'--repeats','1','--seed','1','--timeout','80','--only','solution','--out',str(out)]
            support.write(run/'requested_command.json',{'argv':command,'single_training_hard_cap_s':80,'cap_bypass':False,'scope':'single full64-step diagnostic, not paired scoring'})
            status['status']='RUNNING_STAGE_DIAGNOSTIC_ONLY';support.write(export/'status.json',status)
            # This supervisor covers warm-up + one separately80s-capped training
            # command + native public evaluation. It cannot extend training.
            process=process_guard.run(command,cwd=repo,env=env,log=run/'harness.log',timeout=300,lock_fd=lockfd)
            raw=support.read(out/'result.json') if (out/'result.json').is_file() else {}
            logfile=out/'solution_1/train.log';log=logfile.read_text(errors='replace') if logfile.is_file() else ''
            records=events(log,'SOUP_STAGE_TIMING ');support.write(run/'stage_records.json',{'scope':'host monotonic native-stage records only, not synchronized GPU timing','records':records,'score':None,'admission':'UNKNOWN'})
            errors=diagnostic_errors(raw,process,log);status.update(process=process,errors=errors,complete_native_command_and_eval=not errors,status='COMPLETE_STAGE_DIAGNOSTIC_ONLY' if not errors else 'INCOMPLETE_CENSORED_STAGE_DIAGNOSTIC_ONLY')
            support.verify_official(repo,official)
            if errors:tree.quarantine('Incomplete fixed-task stage diagnostic; no automatic retry')
    except BaseException as exc:
        status.update(status='BLOCKED_FAILED_OR_CENSORED_STAGE_DIAGNOSTIC_ONLY',error=repr(exc),complete_native_command_and_eval=False)
        if lockfd is not None:tree.quarantine('Stage diagnostic failure: '+repr(exc))
    finally:
        status.update(orchestrator_wall_s=time.monotonic()-started,score=None,admission='UNKNOWN',complete_fixed_task=False)
        support.export_small(run,export);status['export_omitted']=dict(support.EXPORT_OMITTED);support.write(export/'status.json',status)
        if lockfd is not None:tree.finish_quarantine();os.close(lockfd)
    print(json.dumps({'status':status['status'],'scope':status['scope'],'export':str(export),'score':None,'admission':'UNKNOWN'}))
    raise SystemExit(0 if status['complete_native_command_and_eval'] else 1)

if __name__=='__main__':main()
