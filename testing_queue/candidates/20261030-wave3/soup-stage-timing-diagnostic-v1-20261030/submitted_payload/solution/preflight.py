"""CPU-only no-op-wrapper gate. Never initializes CUDA or runs training."""
import argparse, json, random
from pathlib import Path
from . import stage_probe as p


def run():
    events=[];old_emit=p.emit;p.emit=lambda *x:events.append(x)
    try:
        class Fixture:
            def call(self, x, *, y): return x, y, random.random()
            def fail(self): raise ValueError('native sentinel')
        f=Fixture();token=object();random.seed(192);state=random.getstate();native=f.call(token,y=3);random.setstate(state)
        assert p.wrap(Fixture,'call','fixture') is True
        assert p.wrap(Fixture,'call','fixture') is False
        actual=f.call(token,y=3);assert actual==native and actual[0] is token
        p.wrap(Fixture,'fail','failure')
        try:f.fail()
        except ValueError as exc:assert str(exc)=='native sentinel'
        else:raise AssertionError('exception swallowed')
        assert events==[('fixture','enter'),('fixture','return'),('failure','enter'),('failure','raise')]
        p.restore();assert not getattr(Fixture.call,'__soup_stage_probe__',False)
        return {'status':'PASS_CPU_OBSERVATIONAL_WRAPPERS_ONLY','gpu_executed':False,
                'training_executed':False,'numerical_optimization':False,'score':None,'admission':'UNKNOWN'}
    finally:p.restore();p.emit=old_emit

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);x=a.parse_args();v=run();Path(x.out).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
