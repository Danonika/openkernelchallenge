import contextlib, io, json, random, sys, types, unittest
from unittest import mock
from solution import stage_probe as p
from solution.preflight import run
class Tests(unittest.TestCase):
 def tearDown(self):p.restore();p._sequence=0;p._origin=None
 def test_cpu_contract(self):self.assertEqual(run()['status'],'PASS_CPU_OBSERVATIONAL_WRAPPERS_ONLY')
 def test_bounded_direct_fd(self):
  with mock.patch.object(p.os,'write') as w:
   for _ in range(100):p.emit('fixed_stage','return')
   self.assertEqual(w.call_count,48)
   for c in w.call_args_list:
    fd,raw=c.args;self.assertEqual(fd,1);self.assertLessEqual(len(raw),1024);self.assertTrue(raw.endswith(b'\n'));v=json.loads(raw[len(p._PREFIX):]);self.assertGreaterEqual(v['since_apply_ns'],0)
 def test_broken_pipe_does_not_alter_native_return(self):
  class F:
   def call(self,x):return x
  p.wrap(F,'call','fixture');token=object()
  with mock.patch.object(p.os,'write',side_effect=BrokenPipeError):self.assertIs(F().call(token),token)
 def test_native_stdout_stays(self):
  class F:
   def call(self):print('native output');return 7
  p.wrap(F,'call','fixture');out=io.StringIO()
  with mock.patch.object(p.os,'write'),contextlib.redirect_stdout(out):self.assertEqual(F().call(),7)
  self.assertEqual(out.getvalue(),'native output\n')
 def test_foreign_patch_not_overwritten_on_restore(self):
  class F:
   def call(self):return 1
  p.wrap(F,'call','fixture');foreign=lambda self:2;F.call=foreign;p.restore();self.assertIs(F.call,foreign)
 def test_baseexception_identity(self):
  exc=KeyboardInterrupt('sentinel')
  class F:
   def call(self):raise exc
  p.wrap(F,'call','fixture')
  with mock.patch.object(p.os,'write'):
   try:F().call()
   except KeyboardInterrupt as caught:self.assertIs(caught,exc)
   else:self.fail('swallowed interruption')
 def test_missing_hook_native_noop(self):self.assertFalse(p.wrap(types.SimpleNamespace(),'missing','absent'))
 def test_install_idempotent_fake_native_modules(self):
  class Wrapper:
   def setup(self):pass
   def _setup_transformers(self):pass
   def train(self):pass
  class Trainer:
   def __init__(self):self.marker=1
   def train(self):return 'native'
   def _save_checkpoint(self):return None
   def save_model(self):return None
  sft=types.ModuleType('soup_cli.trainer.sft');sft.SFTTrainerWrapper=Wrapper;sft._assert_finite_training_state=lambda *a:None
  soup=types.ModuleType('soup_cli');tr=types.ModuleType('soup_cli.trainer');soup.trainer=tr;tr.sft=sft
  hf=types.ModuleType('transformers');hf.Trainer=Trainer
  with mock.patch.dict(sys.modules,{'soup_cli':soup,'soup_cli.trainer':tr,'soup_cli.trainer.sft':sft,'transformers':hf}),mock.patch.object(p.os,'write'),mock.patch.object(p.atexit,'register') as ar:
   p.install();p.install();self.assertEqual(len(p._owned),8);self.assertEqual(ar.call_count,1);self.assertEqual(Trainer().train(),'native')
if __name__=='__main__':unittest.main(verbosity=2)
