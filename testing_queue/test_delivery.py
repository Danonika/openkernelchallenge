"""New client and installer checks only; no package/GPU installation."""
import base64,contextlib,hashlib,io,json,os,tempfile,threading,unittest
from pathlib import Path
from api import LocalServer
from agent_client import Client,ClientError
from install import verify
P=Path(__file__).resolve().parent
TOKENS={role:'TEST_ONLY_DELIVERY_'+role.upper()+'_'+'x'*32 for role in ('submitter','reviewer','worker')}
class DeliveryTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.server=LocalServer(Path(self.t.name)/'state',TOKENS);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_port)
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.t.cleanup()
 def test_client_upload_status_download_review(self):
  reviewer=Client(self.url,TOKENS['reviewer']);submitter=Client(self.url,TOKENS['submitter']);reviewer.request('POST','/v1/baselines',json.loads((P/'examples/baseline_cpu.json').read_text()))
  raw=(P/'examples/candidate.ipynb').read_bytes();receipt=submitter.request('POST','/v1/jobs',{'candidate_id':'CLIENT-SMOKE','backend':'CPU','baseline_id':'base_CPU','entrypoint':'cpu_contract_v1','artifact_kind':'notebook','priority':20,'contains_secrets':False,'artifact_base64':base64.b64encode(raw).decode()})
  self.assertEqual(submitter.request('GET','/v1/jobs/'+receipt['job_id'])['state'],'submitted');out=Path(self.t.name)/'review.ipynb';reviewer.download(receipt['job_id'],out,receipt['source_sha256']);self.assertEqual(out.read_bytes(),raw)
  with self.assertRaises(FileExistsError):reviewer.download(receipt['job_id'],out,receipt['source_sha256'])
  reviewer.request('POST','/v1/jobs/'+receipt['job_id']+'/review',{'source_sha256':receipt['source_sha256'],'approved':True,'authorization_ref':'test-explicit-approval'})
  self.assertEqual(submitter.request('GET','/v1/jobs/'+receipt['job_id'])['state'],'approved')
 def test_client_transport_bounds_and_no_impersonation(self):
  for url in ['http://192.0.2.20','https://example.invalid/redacted','https://example.com/path','https://example.com?token=x']:
   with self.assertRaises(ClientError):Client(url,TOKENS['submitter'])
  c=Client(self.url,TOKENS['submitter'])
  with self.assertRaises(ClientError):c.request('POST','/v1/claims',{})
  with self.assertRaises(ClientError):c.request('GET','/v1/jobs?pass=00009999')
  with self.assertRaises(ClientError):Client(self.url,'00009999')
 def test_installer_hash_and_traversal_fail_closed(self):
  root=Path(self.t.name)/'package';root.mkdir();(root/'a.py').write_text('pass\n');m={'files':{'a.py':hashlib.sha256(b'pass\n').hexdigest()}};(root/'PACKAGE_MANIFEST.json').write_text(json.dumps(m));self.assertEqual(len(verify(root)),1)
  (root/'a.py').write_text('changed')
  with self.assertRaises(ValueError):verify(root)
  m['files']={'../outside':'0'*64};(root/'PACKAGE_MANIFEST.json').write_text(json.dumps(m))
  with self.assertRaises(ValueError):verify(root)
 def test_asset_manifest_is_selected_public_read_only(self):
  from unittest.mock import patch
  import prepare_asset_manifest as assets
  cache=Path(self.t.name)/'cache';repo=Path(self.t.name)/'repo';root=cache/assets.MODEL
  for d in (root/'refs',root/'blobs',root/'snapshots'/assets.REVISION,repo/'data'):d.mkdir(parents=True,exist_ok=True)
  (root/'refs/main').write_text(assets.REVISION);(root/'blobs/publicblob').write_bytes(b'public-model-fixture');link=root/'snapshots'/assets.REVISION/'config.json';link.symlink_to('../../blobs/publicblob')
  data=repo/'data/train.jsonl';data.write_bytes(b'public-data-fixture')
  with patch.object(assets,'DATA',{'data/train.jsonl':assets.sha(data)}):
   manifest,hashes=assets.build(cache,repo);self.assertEqual(len(manifest['files']),2);self.assertEqual(hashes['data/train.jsonl'],assets.sha(data));self.assertTrue(link.is_symlink())
   outside=Path(self.t.name)/'outside';outside.write_text('not selected');link.unlink();link.symlink_to(outside)
   with self.assertRaises(ValueError):assets.build(cache,repo)

 def test_release_is_fixed_allowlist_and_reference_has_no_private_paths(self):
  import build_release
  expected=set(build_release.CORE+build_release.ASSETS)
  self.assertIn('reference/kaggle_environment.json',expected)
  self.assertIn('RUNNER_ADAPTER.md',expected)
  self.assertFalse(any(n.startswith('state/') or n.endswith(('.log','.sqlite3','.env')) for n in expected))
  self.assertNotIn('REPORT.md',expected);self.assertNotIn('integrity.json',expected)
  ref=json.loads((P/'reference/kaggle_environment.json').read_text())
  self.assertTrue(all(len(v)==64 for v in ref['provenance'].values()))
  self.assertNotIn('/workspace/',json.dumps(ref))
  self.assertIsNone(ref['software']['kernel_release'])
  self.assertIsNone(ref['software']['os_distribution'])

 def test_source_bundle_ignores_generated_bytecode_not_secrets(self):
  from bundle_candidate import bundle
  source=Path(self.t.name)/'solution';source.mkdir();(source/'__init__.py').write_text('def queue_smoke(): return {"passed": True}\n')
  first,files=bundle(source);(source/'__pycache__').mkdir();(source/'__pycache__/__init__.cpython-312.pyc').write_bytes(b'not source')
  second,newfiles=bundle(source);self.assertEqual(first,second);self.assertEqual(files,newfiles)
  (source/'.env').write_text('synthetic fixture, never a real secret')
  with self.assertRaises(Exception):bundle(source)

if __name__=='__main__':unittest.main(verbosity=2)
