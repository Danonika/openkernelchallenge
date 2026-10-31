#!/usr/bin/env python3
"""Synthetic reviewer-logic tests; no GPU/training/admission evidence."""
import copy,json,hashlib,tempfile
from pathlib import Path
from checklist import evaluate,load,RULES
cfg=load(RULES);rules=cfg['rules'];T=cfg['thresholds']
facts={x['fact']:{'value':True,'evidence':['synthetic control, not real proof']} for x in rules if x['fact']}
e={'schema_version':1,'quality_scope':'synthetic','facts':facts,'eligible_kernel_names':['own_rms'],'solution_sha256':'synthetic-solution'}
def run(name,sol=False):
 return {'run':name,'exit_code':0,'global_step':64,'wall_s':70 if sol else 80,'gpu_peak_mib':13000 if sol else 14000,'gpu_start_mib':0,
 'metrics':{'cuda_device':'Tesla T4','visible_devices':'0','device_count':1,'alloc_conf':'expandable_segments:True','torch_max_reserved_mib':12000,
 'timed_loop_s':10,'own_kernel_ms':{'own_rms':500},'triton_launches':{'own_rms':64},'triton_files':{'own_rms':'solution/original.py'},'banned_modules_loaded':[]},
 'log':{'loss':[2.0]*64,'entropy':[1.0]*64,'mean_token_accuracy':[0.5]*64,'num_tokens':list(range(1,65))},'rewind':[[[i],[3],[2.0]] for i in range(64)]}
r={'task':{'repeats':2},'runs':{'base':[run('base_1'),run('base_2')],'solution':[run('solution_1',True),run('solution_2',True)]},'evals':{'base':{'exit_code':0,'mean_loss':2.0},'solution':{'exit_code':0,'mean_loss':2.0}},'packages_changed':[], 'summary':{'ratios':{'speedup':80/70,'memory_ratio':14000/13000}}}
c={'result_sha256':'synthetic-result','solution_sha256':'synthetic-solution','checks':[{'check':str(i)+'. synthetic','ok':True} for i in range(1,11)]}
def out(ee=e,rr=r,cc=c):return evaluate(ee,rr,cc,'synthetic-result')
def status(o,k):return next(x['status'] for x in o['rows'] if x['id']==k)
results=[]
def test(name,fn):fn();results.append({'name':name,'status':'PASS'})
def eq(a,b):assert a==b,(a,b)
test('synthetic_complete_fixture_pass',lambda:eq(out()['status'],'PASS'))
test('missing_everything_unknown',lambda:eq(evaluate({})['status'],'UNKNOWN'))
def quality():
 q=copy.deepcopy(r);q['evals']['solution']['mean_loss']=2.012
 eq(status(out(rr=q),'hidden-quality'),'FAIL')
 q['evals']['solution']['mean_loss']=2.10;eq(status(out(rr=q),'hidden-quality'),'FAIL')
 q['evals']['solution']['mean_loss']=2.009;eq(status(out(rr=q),'hidden-quality'),'PASS')
test('half_percent_not_five_percent_loss',quality)
def share():
 q=copy.deepcopy(r);q['runs']['solution'][1]['metrics']['own_kernel_ms']['own_rms']=499
 eq(status(out(rr=q),'own-share'),'FAIL')
 q['runs']['solution'][1]['metrics']['own_kernel_ms']['own_rms']=500
 eq(status(out(rr=q),'own-share'),'PASS')
test('five_percent_share_every_run_boundary',share)
def intersection():
 q=copy.deepcopy(r);q['runs']['solution'][1]['metrics']['triton_launches']={'different':64}
 eq(status(out(rr=q),'own-share'),'FAIL')
test('common_eligible_names_required',intersection)
def mean_relative():
 q=copy.deepcopy(r);q['runs']['solution'][0]['log']['entropy']=[1.04,.96]*32
 eq(status(out(rr=q),'entropy'),'FAIL')
test('per_element_gap_not_equal_means',mean_relative)
def first():
 q=copy.deepcopy(r);q['runs']['solution'][0]['log']['loss'][0]=2.021
 eq(status(out(rr=q),'first-loss'),'FAIL');eq(status(out(rr=q),'loss-curve'),'PASS')
test('first_loss_independent_of_average',first)
def token():
 q=copy.deepcopy(r);q['runs']['solution'][0]['log']['num_tokens'][0]+=1
 eq(status(out(rr=q),'tokens'),'FAIL')
test('token_counts_exact',token)
def rewind():
 q=copy.deepcopy(r);q['runs']['solution'][0]['rewind'][0][0]=[99]
 eq(status(out(rr=q),'rewind'),'FAIL')
test('rewind_row_identity_exact',rewind)
def hidden():
 ee=copy.deepcopy(e);ee['quality_scope']='public'
 eq(status(out(ee=ee),'public-quality'),'PASS');eq(status(out(ee=ee),'hidden-quality'),'UNKNOWN');eq(out(ee=ee)['status'],'UNKNOWN')
test('public_not_hidden_admission',hidden)
def hashguard():
 cc=copy.deepcopy(c);cc['result_sha256']='changed';eq(status(out(cc=cc),'official-checks'),'FAIL')
 cc=copy.deepcopy(c);cc['solution_sha256']='changed';eq(status(out(cc=cc),'official-checks'),'FAIL')
test('both_result_and_solution_hashes',hashguard)
def completion():
 q=copy.deepcopy(r);q['evals']['solution']['exit_code']=1;q['wrapper_exit_code']=0
 eq(status(out(rr=q),'finished'),'FAIL')
test('wrapper_success_not_nested_success',completion)
def cold_memory():
 q=copy.deepcopy(r);q['runs']['base'][0]['gpu_peak_mib']=10000;q['runs']['base'][1]['gpu_peak_mib']=20000
 q['runs']['solution'][0]['gpu_peak_mib']=15000;q['runs']['solution'][1]['gpu_peak_mib']=22001
 q['summary']['ratios']['memory_ratio']=20000/22001
 eq(status(out(rr=q),'memory'),'FAIL')
test('memory_max_not_mean',cold_memory)
def noise():
 q=copy.deepcopy(r)
 for a in q['runs']['solution']:a['wall_s']=80/1.04;a['gpu_peak_mib']=14000/1.04
 q['summary']['ratios']={'speedup':1.04,'memory_ratio':1.04}
 o=out(rr=q);eq(status(o,'time'),'PASS');eq(status(o,'memory'),'PASS');eq(status(o,'noise-floor'),'PASS')
 obs=next(a['observed'] for a in o['rows'] if a['id']=='noise-floor');eq(obs['speed_effective_ratio'],1.0);eq(obs['leaderboard_points'],None)
test('noise_floor_not_admission_or_loss',noise)
def missingfact():
 ee=copy.deepcopy(e);ee['facts']['readme_complete']={'value':True,'evidence':[]}
 eq(status(out(ee=ee),'readme'),'UNKNOWN')
test('bare_assertion_not_evidence',missingfact)
def wronggpu():
 q=copy.deepcopy(r);q['runs']['solution'][0]['metrics']['cuda_device']='A100'
 eq(status(out(rr=q),'clean-gpu'),'FAIL')
test('T4_not_other_GPU',wronggpu)
def finite():
 q=copy.deepcopy(r);q['runs']['solution'][0]['log']['entropy'][0]=float('nan')
 eq(status(out(rr=q),'entropy'),'UNKNOWN')
test('nonfinite_not_pass',finite)
def source():
 ee=copy.deepcopy(e);ee['facts']['original_useful_source']['value']=False
 eq(status(out(ee=ee),'own-source'),'FAIL');eq(status(out(ee=ee),'own-share'),'FAIL')
test('copied_or_unused_source_not_owned',source)
if __name__=='__main__':
 print(json.dumps({'scope':'synthetic reviewer logic only','passed':len(results),'failed':0,'tests':results},indent=2))
