"""Run authored mixed target gate via CPU eligibility/device shims, not target proof."""
from pathlib import Path
from unittest.mock import patch
import json
import torch
from solution.verify_rowmix import run
from solution import materialized_ce as ce
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
device=torch.device
report={'scope':'CPU Torch2.15/Triton3.9 interpreter with CUDA metadata/device shims; not target validation',
        'gpu_executed':False,'training_executed':False}
with patch.object(torch,'device',side_effect=lambda x:device('cpu' if x=='cuda' else x)), \
     patch.object(torch.Tensor,'is_cuda',new=property(lambda _:True)), \
     patch.object(torch.cuda,'get_device_capability',return_value=(7,5)), \
     patch.object(torch.cuda,'get_rng_state',side_effect=torch.get_rng_state):
    run(report)
report['status']='PASS_CPU_SHIMMED_PREFLIGHT_ONLY'
(root/'validation/cpu_shimmed_target_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'pair_count':len(report['row_mixed_pairs']),'fallback_count':len(report['row_mixed_fallbacks']),'gpu_executed':False}))
