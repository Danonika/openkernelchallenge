"""Future opt-in target gates, unexecuted during composition preparation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    output=Path(args.out);details=output.parent/'component_preflight';details.mkdir(exist_ok=False)
    report={'status':'FAIL','gpu_executed':False,'score':None,'admission':'UNKNOWN','components':[]}
    try:
        if os.environ.get('TRITON_INTERPRET'):raise RuntimeError('Production preflight rejects interpreter mode')
        import torch
        if torch.__version__!='2.11.0+cu128' or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='Tesla T4':
            raise RuntimeError('One target T4 and exact Torch build required')
        # Re-run unchanged component gates in separate children so their patch
        # and compiler/cache state is not treated as composition evidence.
        for name,module,out in [('memory','solution.memory_component.verify_memory',details/'memory.json'),
                                ('rotary','solution.rotary_component.gpu_preflight',details/'rotary')]:
            process=subprocess.run([sys.executable,'-B','-m',module,'--out',str(out)],check=False)
            report['components'].append({'name':name,'exit_code':process.returncode,'output':str(out)})
            if process.returncode:raise RuntimeError(name+' component preflight failed')
        from .composition_fixture import run_combined
        report['combined']=[run_combined('cuda',scale=1.0),
                            run_combined('cuda',scale=32768.0,retain=True),
                            run_combined('cuda',scale=1.0,outer_saved_hooks=True)]
        # Constructor/MLP/module-function owners must coexist after all gates.
        from . import apply
        installed=apply()
        if not installed['memory'].get('installed') or not installed['head_installed'] or not installed['rotary_installed']:
            raise RuntimeError('Composite production install was partial')
        report['installation']=installed
        torch.cuda.synchronize()
        report.update(status='PASS_PREFLIGHT_ONLY',gpu_executed=True)
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'output':str(output)},sort_keys=True),flush=True)
    raise SystemExit(report['status']!='PASS_PREFLIGHT_ONLY')

if __name__=='__main__':main()
