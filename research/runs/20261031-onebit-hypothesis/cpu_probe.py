from pathlib import Path
import json,time
import numpy as np
from codec import NF4,analyze,self_check

started=time.perf_counter();rng=np.random.default_rng(20261031)
x=rng.normal(size=(4096,64)).astype(np.float32)
x/=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),1e-30)
codes=np.argmin(np.abs(x[:,:,None]-NF4[None,None,:]),axis=2).astype(np.uint8)
report={'scope':'SYNTHETIC_CPU_CODEC_FEASIBILITY_NOT_MODEL_QUALITY_OR_SPEED',
        'self_check':self_check(),'synthetic_normal_nf4':analyze(codes,4),
        'synthetic_normal_fp16':analyze((x*.02).astype(np.float16),16),
        'limitations':['No actual model or activations','Weight RMS budgets are not loss tolerances',
          'No GPU kernel, training speed, peak VRAM, official admission or target claim',
          'One codec design; not an impossibility result for other quantizers'],
        'elapsed_cpu_s':time.perf_counter()-started}
out=Path(__file__).with_name('cpu_results.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'scope':report['scope'],'checks':report['self_check'],
 'nf4_pure_onebit':report['synthetic_normal_nf4']['pure_onebit'],
 'nf4_adaptive':[{k:r[k] for k in ('block_rms_budget','selected_bits_per_weight','selected_native_reference')} for r in report['synthetic_normal_nf4']['adaptive']],
 'fp16_adaptive':[{k:r[k] for k in ('block_rms_budget','selected_bits_per_weight','selected_native_reference')} for r in report['synthetic_normal_fp16']['adaptive']]}))
