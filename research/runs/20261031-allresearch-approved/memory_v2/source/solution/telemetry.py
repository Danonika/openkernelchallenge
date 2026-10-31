"""Bounded metadata-only ownership receipts; no device queries or metric work."""
from collections import Counter
import json,os
class Telemetry:
    def __init__(self,stream=True):self.stream=stream;self.counts=Counter();self.reasons=Counter()
    def emit(self,event,**fields):
        self.counts[event]+=1
        if fields.get('reason') is not None:self.reasons[str(fields['reason'])]+=1
        count=self.counts[event]
        if not self.stream or count&(count-1):return
        payload=('\nSOUP_STREAMED_CE '+json.dumps({'event':event,'count':count,**fields},sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        while payload:
            count=os.write(1,payload)
            if count<=0:raise OSError('No progress writing streamed CE diagnostics')
            payload=payload[count:]
    def snapshot(self):return {'counts':dict(self.counts),'reasons':dict(self.reasons)}
