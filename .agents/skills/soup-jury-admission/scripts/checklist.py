#!/usr/bin/env python3
"""Read-only evidence triage. Never invokes official checks, GPU, network or publication."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
RULES=Path(__file__).resolve().parents[1]/'references/rules-v1.json'
def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def combine(states):return 'FAIL' if 'FAIL' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'PASS'
def tri(v):return 'PASS' if v is True else 'FAIL' if v is False else 'UNKNOWN'
def fact(e,k):
 facts=e.get('facts',{});f=facts.get(k,{}) if isinstance(facts,dict) else {}
 return tri(f.get('value')) if isinstance(f,dict) and isinstance(f.get('evidence'),list) and f['evidence'] and all(isinstance(x,str) and x.strip() for x in f['evidence']) else 'UNKNOWN'
def load(path):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError,TypeError):return None
def schema_errors(e,cfg):
 errors=[];schema=cfg['evidence_schema']
 for k in schema['required']:
  if k not in e:errors.append('missing '+k)
 if e.get('schema_version')!=1 or isinstance(e.get('schema_version'),bool):errors.append('schema_version must be integer1')
 if e.get('quality_scope') not in schema['properties']['quality_scope']['enum']:errors.append('quality_scope is unknown')
 for k in e:
  if k not in schema['properties']:errors.append('unsupported field '+k)
 for k in ('result_path','checks_path','solution_sha256'):
  if k in e and (not isinstance(e[k],str) or not e[k].strip()):errors.append(k+' must be nonempty string')
 if 'eligible_kernel_names' in e and (not isinstance(e['eligible_kernel_names'],list) or not all(isinstance(x,str) and x for x in e['eligible_kernel_names'])):errors.append('eligible_kernel_names must be string array')
 if 'facts' in e:
  if not isinstance(e['facts'],dict):errors.append('facts must be object')
  else:
   for k,f in e['facts'].items():
    if not isinstance(f,dict) or not isinstance(f.get('value'),bool) or not isinstance(f.get('evidence'),list) or not f['evidence'] or not all(isinstance(x,str) and x.strip() for x in f['evidence']):errors.append('malformed fact '+k)
 return errors
def evaluate(e,result=None,checks=None,result_sha=None):
 e=e if isinstance(e,dict) else {};result=result if isinstance(result,dict) else {};checks=checks if isinstance(checks,dict) else None
 cfg=load(RULES);t=cfg['thresholds'];out={};computed={};errors=schema_errors(e,cfg)
 def emit(i,status,observed=None,missing=None):out[i]={'id':i,'status':status,'observed':observed,'missing_proof':missing if status=='UNKNOWN' else None}
 def limit(i,vals,maxv):
  if vals is None:emit(i,'UNKNOWN',missing='finite comparable evidence')
  else:emit(i,tri(all(v<=maxv for v in vals)),max(vals))
 def rel(a,b):return abs(a-b)/max(abs(b),t['relative_floor'])
 def series(a,b):
  if not isinstance(a,list) or not isinstance(b,list) or not a or len(a)!=len(b) or not all(number(v) for v in a+b):return None
  return sum(rel(v,w) for v,w in zip(a,b))/len(a)
 for rule in cfg['rules']:
  if rule['fact']:emit(rule['id'],fact(e,rule['fact']),(e.get('facts') or {}).get(rule['fact']) if isinstance(e.get('facts',{}),dict) else None,rule['fact']+' with artifact reference')
 runs=(result or {}).get('runs',{})
 base=runs.get('base',[]) if isinstance(runs,dict) else [];sol=runs.get('solution',[]) if isinstance(runs,dict) else []
 if not isinstance(base,list):base=[]
 if not isinstance(sol,list):sol=[]
 raw=base+sol;valid_rows=all(isinstance(r,dict) for r in raw)
 if not valid_rows:base=[];sol=[];raw=[]
 both=bool(base and sol);ref=base[0] if base else {};comp=base[1:]+sol
 evals=(result or {}).get('evals',{});evals=evals if isinstance(evals,dict) else {};evals={k:v for k,v in evals.items() if isinstance(v,dict)}
 def m(r):return r.get('metrics',{}) if isinstance(r.get('metrics'),dict) else {}
 def lg(r):return r.get('log',{}) if isinstance(r.get('log'),dict) else {}
 completion=[]
 for r in raw:completion.append(tri(r['exit_code']==0) if 'exit_code' in r and r['exit_code'] is not None else 'UNKNOWN')
 for side in ('base','solution'):
  v=evals.get(side,{}) if isinstance(evals,dict) else {}
  completion.append(tri(v['exit_code']==0) if isinstance(v,dict) and v.get('exit_code') is not None else 'UNKNOWN')
 task=(result or {}).get('task',{});repeats=task.get('repeats') if isinstance(task,dict) else None
 if both and isinstance(repeats,int) and not isinstance(repeats,bool):completion.append(tri(len(base)==len(sol)==repeats and repeats>=1))
 else:completion.append('UNKNOWN')
 emit('finished',combine(completion),{'base_runs':len(base),'solution_runs':len(sol)},'both sides, expected counts and nested train/eval exits')
 emit('paired',combine([tri(len(base)==len(sol)==t['formal_repeats_each']) if both else 'UNKNOWN',fact(e,'same_session_order')]),{'required_order':cfg['aggregation']['run_order']},'two runs each and same-session order proof')
 eligible=e.get('eligible_kernel_names');source_ok=fact(e,'original_useful_source')
 if isinstance(eligible,list):eligible=[n for n in eligible if isinstance(n,str) and not any(n.startswith(p) for p in cfg['excluded_kernel_name_prefixes'])]
 if both and isinstance(eligible,list) and source_ok=='PASS':
  launched=[]
  for r in sol:
   files=m(r).get('triton_files');counts=m(r).get('triton_launches')
   if not isinstance(files,dict) or not isinstance(counts,dict):launched=None;break
   launched.append({n for n in eligible if number(counts.get(n)) and counts[n]>0 and str(files.get(n,'')).replace('\\','/').startswith('solution/')})
  if launched is not None:
   common=set.intersection(*launched) if launched else set();shares=[]
   for r in sol:
    ms=m(r).get('own_kernel_ms');den=m(r).get('timed_loop_s')
    if not isinstance(ms,dict) or not number(den) or den<=0 or any(not number(ms.get(n)) or ms[n]<0 for n in common):shares=None;break
    shares.append(sum(ms[n] for n in common)/(1000*den))
   if shares is not None:emit('own-share',tri(bool(common) and min(shares)>=t['own_share_min']),{'common_names':sorted(common),'per_run_share':shares,'minimum':min(shares)})
 if 'own-share' not in out:emit('own-share','FAIL' if source_ok=='FAIL' else 'UNKNOWN',missing='reviewed eligible original useful names, normal launches, CUDA-event ms and timed_loop_s in every solution run')
 dep=[fact(e,'dependency_source_review')]
 changed=(result or {}).get('packages_changed');dep.append(tri(not changed) if isinstance(changed,list) else 'UNKNOWN')
 for r in sol:
  x=m(r).get('banned_modules_loaded');dep.append(tri(not x) if isinstance(x,list) else 'UNKNOWN')
 if not sol:dep.append('UNKNOWN')
 emit('dependencies',combine(dep),missing='source import review, all runtime banned lists and package-delta inventory')
 bq=evals.get('base',{}).get('mean_loss') if isinstance(evals,dict) else None;sq=evals.get('solution',{}).get('mean_loss') if isinstance(evals,dict) else None
 q=(sq-bq)/bq if number(bq) and bq>0 and number(sq) else None
 qstatus=tri(q<=t['quality_worsening_max']) if q is not None else 'UNKNOWN'
 qstatus=combine([qstatus,out['finished']['status']])
 scope=e.get('quality_scope')
 emit('public-quality',qstatus if scope=='public' else 'UNKNOWN',{'delta_fraction':q,'scope':scope},'paired public evaluation')
 emit('hidden-quality',qstatus if scope=='hidden' or scope=='synthetic' else 'UNKNOWN',{'delta_fraction':q,'scope':scope},'organizer hidden-row evaluation; public result is not a substitute')
 if both and all(number(r.get(k)) and r[k]>0 for r in raw for k in ('wall_s','gpu_peak_mib')):
  slower=statistics.median(r['wall_s'] for r in sol)/statistics.median(r['wall_s'] for r in base)
  heavier=max(r['gpu_peak_mib'] for r in sol)/max(r['gpu_peak_mib'] for r in base)
  summary=(result or {}).get('summary',{});ratios=summary.get('ratios',{}) if isinstance(summary,dict) else {};ratios=ratios if isinstance(ratios,dict) else {};s=ratios.get('speedup');mem=ratios.get('memory_ratio')
  consistent=tri(rel(1/s,slower)<=t['summary_ratio_relative_tolerance'] and rel(1/mem,heavier)<=t['summary_ratio_relative_tolerance']) if number(s) and s>0 and number(mem) and mem>0 else 'UNKNOWN'
  emit('time',combine([tri(slower<=t['time_growth_max']),consistent,out['finished']['status']]),{'solution_over_base':slower,'summary_consistency':consistent})
  emit('memory',combine([tri(heavier<=t['memory_growth_max']),consistent,out['finished']['status']]),{'solution_over_base':heavier,'aggregation':'maximum, not PDF mean','summary_consistency':consistent})
  computed.update(speedup=1/slower,memory_ratio=1/heavier)
  emit('noise-floor','PASS',{'speed_effective_ratio':1/slower if 1/slower>=t['scoring_ratio_floor'] else 1.0,'memory_effective_ratio':1/heavier if 1/heavier>=t['scoring_ratio_floor'] else 1.0,'meaning':'scoring interpretation only; not admission or loss tolerance','leaderboard_points':None})
 else:
  for k in ('time','memory','noise-floor'):emit(k,'UNKNOWN',missing='paired finite whole-command times and sampled peaks')
 if both:
  tokens=lg(ref).get('num_tokens');states=[]
  for r in comp:
   other=lg(r).get('num_tokens');states.append(tri(other==tokens) if isinstance(tokens,list) and tokens and isinstance(other,list) and other and all(number(v) for v in tokens+other) else 'UNKNOWN')
  emit('tokens',combine(states),missing='same nonempty token sequence in every run')
  for i,key in [('loss-curve','loss'),('entropy','entropy'),('accuracy','mean_token_accuracy')]:
   gaps=[series(lg(r).get(key),lg(ref).get(key)) for r in comp]
   limit(i,gaps if all(v is not None for v in gaps) else None,t['curve_gap_max'])
  gaps=[]
  for r in comp:
   a,b=lg(r).get('loss'),lg(ref).get('loss')
   gaps.append(rel(a[0],b[0]) if isinstance(a,list) and a and isinstance(b,list) and b and number(a[0]) and number(b[0]) else None)
  limit('first-loss',gaps if all(v is not None for v in gaps) else None,t['first_loss_gap_max'])
  rwstates=[];rwgaps=[]
  for r in comp:
   a,b=r.get('rewind'),ref.get('rewind')
   if not isinstance(a,list) or not isinstance(b,list) or not a or len(a)!=len(b) or not all(isinstance(x,list) and len(x)==3 for x in a+b):rwstates.append('UNKNOWN');continue
   rwstates.append(tri(all(x[:2]==y[:2] for x,y in zip(a,b))))
   if not all(isinstance(x[2],list) for x in a+b):rwstates.append('UNKNOWN');continue
   v=series([z for x in a for z in x[2]],[z for x in b for z in x[2]])
   rwstates.append(tri(v<=t['curve_gap_max']) if v is not None else 'UNKNOWN');rwgaps.append(v)
  emit('rewind',combine(rwstates),{'gaps':rwgaps},'matching original row IDs/counts and finite row losses')
 else:
  for k in ('tokens','first-loss','loss-curve','entropy','accuracy','rewind'):emit(k,'UNKNOWN',missing='paired full curves/rewind')
 clean=[]
 for r in raw:
  q=m(r);required=['cuda_device','visible_devices','device_count','alloc_conf','torch_max_reserved_mib']
  if any(k not in q for k in required) or not all(number(r.get(k)) for k in ('gpu_start_mib','gpu_peak_mib')) or not number(q.get('torch_max_reserved_mib')):clean.append('UNKNOWN');continue
  clean.append(tri('T4' in str(q['cuda_device']) and q['visible_devices']==cfg['fixed_task']['visible_devices'] and q['device_count']==cfg['fixed_task']['device_count'] and q['alloc_conf']==cfg['fixed_task']['allocator'] and r['gpu_start_mib']<=t['idle_start_max_mib'] and r['gpu_peak_mib']>=t['peak_reserved_min_ratio']*q['torch_max_reserved_mib']))
 emit('clean-gpu',combine(clean) if clean else 'UNKNOWN',missing='hardware, visibility, idle, allocator, external/internal peak evidence')
 # Exact structures are also checked by the unchanged official suite. Never infer them from good loss.
 taskstate=[fact(e,'fixed_task_review')]
 if both:
  for r in raw:taskstate.append(tri(r['global_step']==t['expected_steps']) if isinstance(r.get('global_step'),int) else 'UNKNOWN')
 else:taskstate.append('UNKNOWN')
 emit('task',combine(taskstate),missing='fixed settings/adapter/dropout/optimizer/lr and64-step evidence')
 official='UNKNOWN'
 if isinstance(checks,dict) and result_sha:
  rows=checks.get('checks');bound=checks.get('result_sha256')==result_sha
  supplied_solution=e.get('solution_sha256');solution_bound=checks.get('solution_sha256')==supplied_solution if supplied_solution else None
  if not bound or solution_bound is False:official='FAIL'
  elif isinstance(rows,list) and len(rows)==t['expected_official_checks'] and all(isinstance(x,dict) for x in rows):
   official=combine([tri(x.get('ok')) for x in rows]+[tri(solution_bound)])
   # Task/output structural contradiction remains a contradiction even if a manual fact says true.
   c5=next((x for x in rows if str(x.get('check','')).startswith('5.')),None)
   if c5 and c5.get('ok') is False:out['task']['status']='FAIL'
 emit('official-checks',official,missing='all10 unchanged checks and matching result+solution hashes')
 # All mandatory rows remain visible; public diagnostics/noise-floor do not substitute for admission.
 ordered=[]
 for rule in cfg['rules']:
  row=out.get(rule['id'],{'id':rule['id'],'status':'UNKNOWN','observed':None,'missing_proof':rule['requirement']})
  ordered.append({**row,'group':rule['group'],'requirement':rule['requirement'],'source':rule['source']})
 mandatory=[r['status'] for r in ordered if r['group'] not in ('diagnostic','scoring')]
 return {'schema_version':1,'rule_version':cfg['rule_version'],'status':combine(mandatory+(['UNKNOWN'] if errors else [])),'schema_errors':errors,'interpretation':'Evidence checklist only; not an official admission decision or execution/publication authorization','scope':scope,'rule_source_sha256':sha(RULES),'result_sha256':result_sha,'computed':computed,'aggregation_discrepancy':cfg['aggregation']['discrepancy'],'rows':ordered,'counts':{s:sum(r['status']==s for r in ordered) for s in ('PASS','FAIL','UNKNOWN')}}
def main():
 p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('--out');p.add_argument('--markdown');a=p.parse_args();e=load(a.evidence)
 if not isinstance(e,dict):raise SystemExit('Evidence must be a JSON object')
 result=load(e.get('result_path'));checks=load(e.get('checks_path'))
 h=sha(e['result_path']) if result is not None else None
 report=evaluate(e,result,checks,h);text=json.dumps(report,indent=2)+'\n'
 if a.out:Path(a.out).write_text(text)
 else:print(text,end='')
 if a.markdown:
  lines=['# Soup jury evidence checklist',f"Rules: {report['rule_version']} · status: **{report['status']}** (not official admission)",'',report['aggregation_discrepancy'],'','| Rule | Status | Missing proof / observation |','|---|---|---|']
  for row in report['rows']:
   detail=row['missing_proof'] or str(row['observed']);lines.append('| '+row['id']+' | '+row['status']+' | '+detail.replace('|','/').replace('\n',' ')+' |')
  Path(a.markdown).write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
