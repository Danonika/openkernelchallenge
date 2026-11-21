#!/usr/bin/env python3
"""Read-only packaged-evidence verification, using the Python standard library."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for name,digest in m['files'].items():
 assert sha(ROOT/name)==digest,name
c=load(HERE/'catalog.json')
rows=[dict(c['defaults'],**r) for r in c['rows']]
assert len(rows)==121==len({r['id'] for r in rows})
assert sum(r['original104'] for r in rows)==104
assert all(r['measured_soup_t4_gain'] is None for r in rows)
table=load(HERE/'compact_table.json')
assert len(table)==122 and {r[0] for r in table[1:]}=={r['id'] for r in rows}
assert len({r['bucket'] for r in rows})==5
for r in rows:
 for a in r['cpu_artifacts']:
  assert sha(HERE/a['path'])==a['sha256'],a['path']
passed=0
for sub,script,count in [('', 'run_cpu_semantics.py',8),('cycle2','run_cycle2.py',6),('cycle2b','run_cycle2b.py',3)]:
 p=HERE/'cpu_tests'/sub;s=load(p/'summary.json');result=load(p/'results.json')
 assert s['summary']=={'passed':count,'failed':0}==result['summary']
 assert s['script_sha256']==sha(p/script)==result['script_sha256']
 assert s['environment']['cuda_available'] is False
 assert len(result['results'])==count
 passed+=count
assert passed==17
gdir=ROOT/'graph/baseline';g=load(gdir/'graph.json')
assert len(g['configurations'])==1 and g['configurations'][0]['kind']=='baseline'
b=g['configurations'][0]
assert b['payload']['idea_ids']==[] and b['payload']['keys']==[] and b['payload']['solution_apply'] is False
canonical=json.dumps(b['payload'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
assert hashlib.sha256(canonical.encode()).hexdigest()==b['content_sha256']
for k in ['edges','runs','comparisons','gate_evaluations','conflicts']:assert g[k]==[]
a=g['reference_anchor'];assert a['x']==a['y']==0 and a['z'] is None and a['measured'] is False
for k in ['actual_wall_s','actual_peak_mib','actual_eval_loss']:assert a[k] is None
script=(gdir/'data.js').read_text().strip();assert json.loads(script.removeprefix('window.SOUP_BASELINE = ').removesuffix(';'))==g
html=(gdir/'index.html').read_text();assert html.count('id="baselinePoint"')==1
for marker in ['id="save"','id="ideaList"','id="union"']:assert marker not in html
print(f"PASS: {len(m['files'])} file hashes; 121 IDs (104 original); 17 CPU groups; baseline-only graph. No GPU result.")
