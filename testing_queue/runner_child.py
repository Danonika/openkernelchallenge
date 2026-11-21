#!/usr/bin/env python3
"""Runner-owned child launcher. Not a security sandbox; reviewed trusted source only."""
import argparse, importlib, json, os, runpy, sys, traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cpus',required=True)
    sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('smoke'); s.add_argument('--root',required=True); s.add_argument('--module',required=True); s.add_argument('--out',required=True)
    e=sub.add_parser('environment'); e.add_argument('--checker',required=True); e.add_argument('--repo'); e.add_argument('--out',required=True)
    h=sub.add_parser('harness'); h.add_argument('--repo',required=True); h.add_argument('--out',required=True); h.add_argument('--side',choices=['base','solution'],required=True); h.add_argument('--timeout',type=int,required=True)
    a=p.parse_args(); cpus=[int(x) for x in a.cpus.split(',')]
    os.sched_setaffinity(0,cpus)
    if a.mode=='harness':
        os.chdir(a.repo)
        os.execv(sys.executable,[sys.executable,'-B',str(Path(a.repo)/'harness/run_task.py'),'--only',a.side,'--repeats','1','--seed','1','--timeout',str(a.timeout),'--out',a.out])
    if a.mode=='environment':
        sys.argv=[a.checker,'--gpu','--require-four-cpus','--out',a.out]+(['--repo',a.repo] if a.repo else [])
        runpy.run_path(a.checker,run_name='__main__'); return
    result={'status':'failed','scope':'CPU_IMPORT_AND_CALL_SMOKE','is_training':False}
    try:
        sys.path.insert(0,str(Path(a.root).resolve()))
        module=importlib.import_module(a.module)
        if not Path(module.__file__).resolve().is_relative_to(Path(a.root).resolve()):raise ValueError('Module was not loaded from reviewed staged bundle')
        value=module.queue_smoke()
        if not isinstance(value,dict) or value.get('passed') is not True:
            raise ValueError('Reviewed queue_smoke() must return a dictionary with passed=True')

        json.dumps(value,allow_nan=False)
        result.update(status='passed',candidate_report=value,module_file=str(Path(module.__file__).resolve()))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result['status']=='passed' else 1)
if __name__=='__main__':main()
