#!/usr/bin/env python3
"""Create a source-only delivery ZIP from an explicit allowlist; no DB/secrets/raw logs."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE=[
 'sheets_token_auth.py','test_sheets_token_auth.py',
 'notebook_reference.py','public-baseline.json',
 'notebook_metrics.py','test_notebook_metrics.py',
 'notebook_failfast.py','training_run_limit.py','notebook_sitecustomize.py','test_training_run_limit.py','test_notebook_failfast.py',
 'test_notebook_browser.py',
 'NOTEBOOK_RUNNER.md','HTTPS_RELAY.md','SHEETS_QUEUE.md','sheets_queue_worker.py','test_sheets_queue_worker.py',
 'sheets_results.py','test_sheets_results.py','sheets_result_publisher.py','test_sheets_result_publisher.py',
 'notebook_queue.py','notebook_runner.py','notebook_worker.py','notebook_client.py','test_notebooks.py',
 'README.md','FULL_INSTRUCTIONS.md','HTTP_API.md','DEPLOYMENT_ADAPTER.md','DELIVERY_REPORT.md','RUNNER_ADAPTER.md',
 'soup_queue.py','api.py','agent_client.py','runner_adapter.py','runner_child.py','container_selftest.py',
 'bundle_candidate.py','smoke_demo.py','install.py','build_release.py','check_kaggle_environment.py','prepare_asset_manifest.py','prepare_public_model.py','requirements-kaggle.txt','runner_contract.json',
 'test_web_state.js','test_web_browser.py','run_tests.py','run_api_tests.py','test_queue_unit.py','test_api_unit.py','test_delivery.py','test_runner_adapter.py',
]
ASSETS=[
 'deploy/install_sheets_token.py','deploy/soup-sheets-queue-token.service.example','deploy/to-run-view.json','recovery/soup_sheets_bridge.py','recovery/test_soup_sheets_bridge.py',
 'recovery/create_one_t4.py','recovery/test_create_one_t4.py','recovery/README.md','recovery/restore_public_ip.sh',
 'examples/notebook-storage-probe.ipynb',
 'web/notebooks.css',
 'web/notebooks.html','web/notebooks.js','examples/notebook-gpu-smoke.ipynb','examples/notebook-training-template.ipynb',
 'web/dashboard.css','web/dashboard.html','web/dashboard.js',
 'deploy/Dockerfile','deploy/init_secrets.py','deploy/install_environment.sh',
 'deploy/authorize-sheets-writer.sh',
 'deploy/quiesce-sheets-keyless.py',
 'deploy/nginx_acme_only.conf.template','deploy/nginx_ip.conf.template','deploy/nginx_hostname.conf.example','deploy/render_templates.py',
 'deploy/soup-api.service.template','deploy/soup-certbot-renew.service.template','deploy/soup-certbot-renew.timer.template','deploy/soup-reload-nginx','deploy/soup-https-relay.service.example','deploy/soup-sheets-queue.service.example',
 'smoke/solution/__init__.py',
 'examples/baseline_cpu.json','examples/baseline_gcp.json','examples/baseline_kaggle_t4.json',
 'examples/candidate.ipynb','examples/http_submission.json','examples/result_aborted79.json',
 'examples/result_cpu_completed.json','examples/submission.json','examples/worker_gcp.json',
 'reference/kaggle_environment.json','reference/kaggle_environment.md',
]
def allowed_files():
 # Do not recursively package directories: newly created files need explicit disclosure review.
 files=sorted(set(CORE+ASSETS))
 for name in files:
  path=ROOT/name
  if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
   raise ValueError('Required source missing/linked: '+name)
 return files
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--manifest-out',required=True);a=p.parse_args();out=Path(a.out);manifest_path=Path(a.manifest_out)
 if out.exists() or manifest_path.exists():raise SystemExit('Output exists; choose new output names. Nothing overwritten/deleted.')
 files=allowed_files();snapshot={name:(ROOT/name).read_bytes() for name in files};manifest={'schema_version':1,'scope':'Queue code/tests/docs and operator-selected public baseline summary; no credentials, live state or raw logs','files':{name:hashlib.sha256(snapshot[name]).hexdigest() for name in files}}
 data=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in files:
   entry=zipfile.ZipInfo('soup-test-queue/'+name,(2026,1,1,0,0,0));entry.external_attr=0o100644<<16;entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,snapshot[name])
  entry=zipfile.ZipInfo('soup-test-queue/PACKAGE_MANIFEST.json',(2026,1,1,0,0,0));entry.external_attr=0o100644<<16;entry.compress_type=zipfile.ZIP_DEFLATED;z.writestr(entry,data)
 with manifest_path.open('xb') as f:f.write(data)
 print(json.dumps({'zip':str(out),'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'manifest':str(manifest_path),'manifest_sha256':hashlib.sha256(data).hexdigest(),'files':len(files)},indent=2))
if __name__=='__main__':main()
