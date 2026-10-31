#!/usr/bin/env python3
"""Read-only cycle4 exact reconstruction, including the genuine retained CPU FAIL."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261030-cycle2';C3=HERE.parent/'20261030-cycle3'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for name,digest in m['files'].items():assert sha(HERE/name)==digest,name
c=load(BASE/'catalog.json');rows=[dict(c['defaults'],**{k:v for k,v in r.items() if k!='bucket'}) for r in c['rows']]
for folder,cycle,count in [(C3,3,10),(HERE,4,6)]:
 patch=load(folder/f'jury_cycle{cycle}_canonical_rows.json');changes={r['id']:r for r in patch['changed_rows']}
 assert not patch['new_ids'] and len(changes)==count and set(changes).issubset(r['id'] for r in rows)
 rows=[changes.get(r['id'],r) for r in rows]
assert len(rows)==121==len({r['id'] for r in rows}) and sum(r['original104'] for r in rows)==104
assert all(r['measured_soup_t4_gain'] is None for r in rows)
assert hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['reconstructed_rows_sha256']
for r in rows:
 for a in r['cpu_artifacts']:
  assert not Path(a['path']).is_absolute()
  matches=[p/a['path'] for p in [HERE,C3,BASE] if (p/a['path']).exists()]
  assert matches and sha(matches[0])==a['sha256'],a['path']
p=HERE/'cpu_tests/cycle4';h=load(p/'head_results.json');n=load(p/'nf4_results.json');s=load(p/'summary.json')
assert h['summary']=={'passed':4,'failed':0} and n['summary']=={'passed':3,'failed':1}
assert (s['passed'],s['failed'])==(7,1) and len(h['results'])==4 and len(n['results'])==4
assert sha(p/'head_oracle.py')==h['script_sha256'] and sha(p/'nf4_oracle.py')==n['script_sha256']
failures=[r for r in n['results'] if r['status']=='FAIL'];assert len(failures)==1 and failures[0]['name']=='actual_cpu_row_alignment_parity_probe'
print('PASS: exact121 rows/104 originals; six replacements; cycle4 7 PASS + 1 retained genuine CPU FAIL; cumulative31 evidence groups, no candidate GPU claim.')
