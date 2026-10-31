from pathlib import Path
import ast, hashlib, json, math
R=Path(__file__).resolve().parent
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'manifest.json').read_text())
for p,s in m['files'].items(): assert sha(R/p)==s,p
h=json.loads((R/'trial001_verified_handoff.json').read_text())
for p,s in h['artifact_sha256'].items(): assert sha(R/p)==s,p
for p,s in h['source_sha256'].items(): assert sha(R/p)==s,p
sm=json.loads((R/'source_manifest.json').read_text())
assert sha(R/'kaggle_runner.py')==sm['runner_sha256']
for p,s in sm['files'].items(): assert sha(R/p)==s,p
c=json.loads((R/'runs_trial001/checks.json').read_text())
assert c['result_sha256']==sha(R/'runs_trial001/result.json')==h['result_sha256']
assert c['solution_sha256']==h['solution_folder_sha256']
assert len(c['checks'])==10 and sum(x['ok'] for x in c['checks'])==9
failed=[x for x in c['checks'] if not x['ok']]
assert len(failed)==1 and '5%' in failed[0]['check']
shares=[x['eligible_own_share_fraction'] for x in h['observed_runs'] if x['side']=='solution']
assert math.isclose(min(shares),0.012052489760724294) and min(shares)<0.05
result=json.loads((R/'runs_trial001/result.json').read_text())
assert result['summary']==h['summary']
assert h['full_admission'] is False and h['eligible_for_points'] is False
assert h['artifact_completeness']['complete_archive'] is False
for p in R.rglob('*.py'): ast.parse(p.read_text())
print(json.dumps({'files_verified':len(m['files']),'official_checks_pass':9,'official_checks_fail':1,'own_share_fraction':min(shares),'standalone_admitted':False,'score_points':None,'full_archive':False}))
