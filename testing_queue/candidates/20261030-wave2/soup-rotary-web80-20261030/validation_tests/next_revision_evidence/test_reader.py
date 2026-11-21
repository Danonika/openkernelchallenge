"""Source-bound real log plus synthetic parser-only checks; no job rerun."""
from pathlib import Path
import hashlib,json
from evidence_reader import events
root=Path(__file__).resolve().parent
cases=[
 ('plain','EVENT {"count":64}',[{'count':64}]),
 ('ansi','\x1b[2K\x1b[1AEVENT {"count":64}\n',[{'count':64}]),
 ('wrapped','EVENT {"count":64,\n"event":\n"done"}\n',[{'count':64,'event':'done'}]),
 ('crlf','EVENT {"count":\r\n64}\r\n',[{'count':64}]),
 ('unanchored','not EVENT {"count":64}',[]),
 ('malformed','EVENT {"count":64',[]),
 ('scalar','EVENT 64',[]),
 ('trailing_junk','EVENT {"count":64} JUNK',[]),
 ('duplicate','EVENT {"count":1,"count":64}',[]),
 ('nonfinite','EVENT {"count":NaN}',[]),
 ('wrong_prefix','OTHER {"count":64}',[]),
 ('osc','\x1b]0;title\x07EVENT {"count":64}',[{'count':64}]),
 ('two_events','EVENT {"count":1}\nEVENT {"count":2}',[{'count':1},{'count':2}]),
]
for name,text,wanted in cases:assert list(events(text,'EVENT '))==wanted,name
assert list(events('EVENT '+json.dumps({'value':'x'*100}),'EVENT ',max_event_chars=32))==[]
log=Path('/workspace/shared/soup_head_research_20261030/evidence/fusion_b_solution_1.log')
binding=json.loads(log.with_name('fusion_b_binding.json').read_text())
assert hashlib.sha256(log.read_bytes()).hexdigest()==binding['member_sha256']
text=log.read_text();counts={}
for prefix in ('SOUP_HEAD_NATIVE ','ROTARY_EVIDENCE '):
 old=[]
 for line in text.splitlines():
  if not line.startswith(prefix):continue
  try:record=json.loads(line[len(prefix):])
  except (ValueError,TypeError):continue
  if isinstance(record,dict):old.append(record)
 new=list(events(text,prefix));assert len(new)>len(old)
 counts[prefix]={'old_single_line_records':len(old),'proposed_records':len(new),'old_events':sorted({r.get('event','') for r in old}),'proposed_events':sorted({r.get('event','') for r in new}),'max_counts':{event:max(r.get('count',0) for r in new if r.get('event')==event) for event in sorted({r.get('event','') for r in new})}}
assert 'forward_launch_returned' in counts['ROTARY_EVIDENCE ']['proposed_events']
assert 'forward_launch_returned' not in counts['ROTARY_EVIDENCE ']['old_events']
report={'status':'PARSER_ONLY_CPU_PASS','synthetic_checks':len(cases)+1,'actual_log_binding':binding,'counts':counts,'reader_sha256':hashlib.sha256((root/'evidence_reader.py').read_bytes()).hexdigest(),'does_not_reclassify_timed_out_run':True,'source_job_actual_status':'solution_1 exceeded80 seconds; incomplete/censored; no comparison/score','queued_artifacts_modified':False,'gpu_or_auth_or_submission':False}
(root/'parser_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
