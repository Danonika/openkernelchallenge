"""Cumulative bounded own diagnostics; native Soup output is untouched."""
from collections import Counter
import json,os
class Telemetry:
    def __init__(self,stream=True):self.stream=stream;self.counts=Counter();self.reasons=Counter()
    def emit(self,event,**fields):
        self.counts[event]+=1
        if 'reason' in fields:self.reasons[fields['reason']]+=1
        n=self.counts[event]
        if not self.stream or n&(n-1):return
        raw=('\nSOUP_HEAD_SHARED '+json.dumps({'event':event,'count':n,**fields},sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        while raw:
            wrote=os.write(1,raw)
            if wrote<=0:raise OSError('No progress writing head diagnostics')
            raw=raw[wrote:]
    def snapshot(self):return {'counts':dict(self.counts),'reasons':dict(self.reasons),'original_triton_kernels':[],'own_triton_share':0.0}
