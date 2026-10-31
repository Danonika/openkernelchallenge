#!/usr/bin/env python3
"""Read-only verification of the sanitized baseline source package."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=load(ROOT/'manifest.json')
for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
g=load(ROOT/'graph.json');assert len(g['configurations'])==1 and len(g['runs'])==1
b=g['configurations'][0];r=g['runs'][0];a=g['reference_anchor']
assert b['kind']=='baseline' and not b['payload']['idea_ids'] and not b['payload']['keys'] and b['payload']['solution_apply'] is False
canonical=json.dumps(b['payload'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
assert hashlib.sha256(canonical).hexdigest()==b['content_sha256']
assert r['role']=='base' and r['status']=='completed' and r['repeats']==1 and r['config_id']==b['id']
assert (r['wall_s'],r['gpu_peak_mib'],r['eval_loss'],r['steps'])==(77.92,14345,2.04965,64)
assert r['admission_status']=='unknown' and r['official_points'] is None
assert (a['x'],a['y'],a['z'])==(0,0,None) and a['measured'] is True
assert all(g[k]==[] for k in ['edges','comparisons','gate_evaluations','conflicts'])
for e in g['evidence']:
 if e['uri'].startswith('evidence/'):assert sha(ROOT/e['uri'])==e['sha256']
script=(ROOT/'data.js').read_text().strip();assert json.loads(script.removeprefix('window.SOUP_BASELINE = ').removesuffix(';'))==g
assert (ROOT/'index.html').read_text().count('id="baselinePoint"')==1
for f in ROOT.rglob('*'):
 if f.is_file() and f.name!='verify_snapshot.py':
  text=f.read_text();assert '/workspace/' not in text and 'kaggle.com/code/' not in text
assert g['environments'][0]['session_id']=='redacted-private-session'
assert all(k not in r for k in ['kernel_url','kernel_id'])
print('PASS: exact sanitized source/artifact hashes; one B0 + one provisional base run; no candidates, private notebook URL/ID, official points or admission.')
