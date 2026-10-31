#!/usr/bin/env python3
"""Read-only exact cycle3 reconstruction and evidence verification."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261030-cycle2'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for name,digest in m['files'].items():assert sha(HERE/name)==digest,name
c=load(BASE/'catalog.json');old=[dict(c['defaults'],**{k:v for k,v in r.items() if k!='bucket'}) for r in c['rows']]
p=load(HERE/'jury_cycle3_canonical_rows.json');patch={r['id']:r for r in p['changed_rows']}
assert not p['new_ids'] and len(patch)==10 and set(patch).issubset(r['id'] for r in old)
rows=[patch.get(r['id'],r) for r in old]
assert len(rows)==121==len({r['id'] for r in rows}) and sum(r['original104'] for r in rows)==104
assert all(r['measured_soup_t4_gain'] is None for r in rows)
canonical=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
assert hashlib.sha256(canonical).hexdigest()==m['reconstructed_rows_sha256']
for r in rows:
 for a in r['cpu_artifacts']:
  path=HERE/a['path']
  if not path.exists():path=BASE/a['path']
  assert sha(path)==a['sha256'],a['path']
s=load(HERE/'cpu_tests/cycle3/summary.json');r=load(HERE/'cpu_tests/cycle3/results.json')
assert s['summary']=={'passed':6,'failed':0}==r['summary']
assert len(r['results'])==6 and s['environment']['cuda_available'] is False
assert sha(HERE/'cpu_tests/cycle3/run_cycle3.py')==s['script_sha256']==r['script_sha256']
print('PASS: immutable cycle3 files; 121 exact reconstructed rows, 104 originals, 10 replacements, 6 new CPU groups (23 total). No candidate GPU result.')
