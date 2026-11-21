"""Bounded metadata-only diagnostics for common consumer ownership."""
from collections import Counter
import json
from .producer.telemetry import emit_line
class Telemetry:
    def __init__(self,stream=True):self.stream=stream;self.counts=Counter();self.reasons=Counter()
    def emit(self,event,**fields):
        self.counts[event]+=1
        if 'reason' in fields:self.reasons[fields['reason']]+=1
        n=self.counts[event]
        if not self.stream or (event!='native_fallback' and n&(n-1)):return
        emit_line('SOUP_CE_METRIC '+json.dumps({'event':event,'count':n,**fields},sort_keys=True,separators=(',',':'),allow_nan=False))
    def snapshot(self):return {'counts':dict(self.counts),'reasons':dict(self.reasons)}
