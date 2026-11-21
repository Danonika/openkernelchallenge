#!/usr/bin/env python3
"""Durable local coordination only. Never executes submitted artifacts/commands."""
import argparse,ast,contextlib,hashlib,io,json,math,os,re,sqlite3,stat,tempfile,time,uuid,zipfile
from pathlib import Path,PurePosixPath
BACKENDS={'CPU','KAGGLE_T4','GCP'}
ENTRYPOINTS={'cpu_contract_v1','gpu_preflight_v1','soup_paired_v1','soup_screen_v1'}
MAX_ARTIFACT=8*1024*1024;MAX_EXPANDED=32*1024*1024;MAX_FILES=128
IDENT=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z');SHA=re.compile(r'[0-9a-f]{64}\Z');COMMIT=re.compile(r'[0-9a-f]{40}\Z')
class QueueError(ValueError):pass

def require(ok,message):
 if not ok:raise QueueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(data):return hashlib.sha256(data).hexdigest()
def identifier(x):return isinstance(x,str) and IDENT.fullmatch(x) is not None
def sha(x):return isinstance(x,str) and SHA.fullmatch(x) is not None
def fields(obj,required,optional=()):
 require(type(obj) is dict,'Expected JSON object');require(set(required)<=set(obj),'Missing fields: '+str(sorted(set(required)-set(obj))));require(set(obj)<=set(required)|set(optional),'Unknown fields (commands/options are not accepted): '+str(sorted(set(obj)-set(required)-set(optional))))
def safe_relative(value):
 require(isinstance(value,str) and 0<len(value)<=240 and re.fullmatch(r'[A-Za-z0-9_. /-]+',value),'Invalid artifact path characters')
 p=PurePosixPath(value);require(not p.is_absolute() and str(p)==value and all(x not in ('','.','..') for x in p.parts),'Absolute/traversal/noncanonical path rejected');return p

def notebook(raw):
 require(len(raw)<=MAX_ARTIFACT,'Notebook too large')
 try:n=json.loads(raw)
 except Exception as e:raise QueueError('Invalid notebook JSON') from e
 require(type(n) is dict and n.get('nbformat')==4 and type(n.get('cells')) is list,'Only notebook v4 data accepted')
 for cell in n['cells']:
  require(type(cell) is dict and cell.get('cell_type') in ('code','markdown','raw'),'Invalid notebook cell')
  if cell['cell_type']!='code':continue
  require(not cell.get('outputs') and cell.get('execution_count') is None,'Notebook outputs/execution state must be cleared before submission')
  source=cell.get('source','');source=''.join(source) if isinstance(source,list) and all(isinstance(x,str) for x in source) else source
  require(isinstance(source,str),'Invalid cell source')
  require(not re.search(r'^\s*[!%]',source,re.M),'Shell/magic notebook cells are not executable queue inputs')
  try:tree=ast.parse(source)
  except SyntaxError as e:raise QueueError('Notebook code must be plain parseable Python for manual review') from e
  for node in ast.walk(tree):
   if not isinstance(node,ast.Call):continue
   name=ast.unparse(node.func)
   require(name not in {'eval','exec','compile','__import__','os.system','os.popen','subprocess.run','subprocess.call','subprocess.Popen','subprocess.check_call','subprocess.check_output','get_ipython'},'Obvious executable shell/dynamic notebook instruction rejected: '+name)

 return {'notebook_cells':len(n['cells']),'review_required':True}

def validate_artifact(raw,kind):
 require(0<len(raw)<=MAX_ARTIFACT,'Artifact empty or over8MiB')
 if kind=='notebook':return notebook(raw)
 require(kind=='solution_bundle','Unknown artifact kind')
 try:z=zipfile.ZipFile(io.BytesIO(raw))
 except zipfile.BadZipFile as e:raise QueueError('Solution bundle must be ZIP data') from e
 names=set();total=0;count=0
 require(len(z.infolist())<=MAX_FILES*2,'Too many ZIP members')
 with z:
  for entry in z.infolist():
   name=entry.filename.rstrip('/') if entry.is_dir() else entry.filename;p=safe_relative(name)
   require(name not in names,'Duplicate ZIP member');names.add(name)
   require(not(entry.flag_bits&1),'Encrypted ZIP rejected')
   mode=entry.external_attr>>16;require(not stat.S_ISLNK(mode),'ZIP symlink rejected')
   require(not any(x.lower() in {'.git','.aws','.ssh','__pycache__','.env','tokens','credentials','adapters','optimizer','triton_cache','inductor_cache'} for x in p.parts),'Forbidden secret/state/cache path')
   if entry.is_dir():continue
   count+=1;total+=entry.file_size;require(count<=MAX_FILES and total<=MAX_EXPANDED,'Bundle bounds exceeded')
   require(p.suffix.lower() in {'.py','.ipynb','.json','.md','.txt','.toml','.yaml','.yml'},'Only reviewable source/text artifacts accepted; no weights/binaries/nestedarchives')
   data=z.read(entry);require(len(data)==entry.file_size,'ZIP size mismatch')
   if p.suffix.lower()=='.ipynb':notebook(data)
   else:
    try:data.decode('utf-8')
    except UnicodeDecodeError as e:raise QueueError('Bundle text must be UTF-8') from e
 require(count>0,'Empty source bundle')
 return {'files':count,'expanded_bytes':total,'review_required':True}

def finite_number(x):return type(x) in (int,float) and math.isfinite(x) and x>=0

class Queue:
 def __init__(self,root,clock=time.time):
  self.root=Path(root).absolute();self.clock=clock
  self.root.mkdir(parents=True,exist_ok=True)
  for p in [self.root,*self.root.parents]:require(not p.is_symlink(),'Queue root symlink rejected')
  for n in ('incoming','artifacts'):
   p=self.root/n;require(not p.is_symlink(),'Internal symlink rejected');p.mkdir(exist_ok=True)
  self.db=self.root/'queue.sqlite3';require(not self.db.is_symlink(),'Database symlink rejected')
  with contextlib.closing(self.connect()) as c:
   c.executescript('''CREATE TABLE IF NOT EXISTS baselines(id TEXT PRIMARY KEY,backend TEXT NOT NULL,body TEXT NOT NULL,hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,dedup TEXT UNIQUE NOT NULL,body TEXT NOT NULL,priority INTEGER NOT NULL,state TEXT NOT NULL,created REAL NOT NULL,review TEXT,attempt TEXT,result TEXT,result_hash TEXT);
CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,job TEXT NOT NULL REFERENCES jobs(id),runner TEXT NOT NULL,backend TEXT NOT NULL,resource TEXT NOT NULL,state TEXT NOT NULL,claimed REAL NOT NULL,lease_until REAL NOT NULL,started REAL,run_reference TEXT);
CREATE TABLE IF NOT EXISTS locks(backend TEXT NOT NULL,resource TEXT NOT NULL,attempt TEXT UNIQUE NOT NULL REFERENCES attempts(id),PRIMARY KEY(backend,resource));
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,at REAL NOT NULL,kind TEXT NOT NULL,body TEXT NOT NULL);''')
 def connect(self):
  require(not self.db.is_symlink(),'Database symlink rejected')
  c=sqlite3.connect(self.db,timeout=10,isolation_level=None);c.row_factory=sqlite3.Row;c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL');return c
 @contextlib.contextmanager
 def tx(self):
  c=self.connect()
  try:c.execute('BEGIN IMMEDIATE');yield c;c.commit()
  except Exception:c.rollback();raise
  finally:c.close()
 def event(self,c,job,kind,body):c.execute('INSERT INTO events(job,at,kind,body) VALUES(?,?,?,?)',(job,self.clock(),kind,canonical(body)))
 def register_baseline(self,spec):
  fields(spec,{'baseline_id','backend','environment_fingerprint','task_fingerprint','harness_commit','evidence_sha256'})
  require(identifier(spec['baseline_id']) and spec['backend'] in BACKENDS,'Invalid baseline identity/backend')
  require(all(sha(spec[k]) for k in ('environment_fingerprint','task_fingerprint','evidence_sha256')) and isinstance(spec['harness_commit'],str) and COMMIT.fullmatch(spec['harness_commit']),'Invalid baseline provenance hashes')
  body=canonical(spec);h=digest(body.encode())
  with self.tx() as c:
   old=c.execute('SELECT * FROM baselines WHERE id=?',(spec['baseline_id'],)).fetchone()
   if old:require(old['body']==body,'Baseline IDs are immutable; use a new ID');return {'baseline_id':spec['baseline_id'],'deduplicated':True,'hash':h}
   c.execute('INSERT INTO baselines VALUES(?,?,?,?)',(spec['baseline_id'],spec['backend'],body,h));self.event(c,None,'baseline_registered',spec)
  return {'baseline_id':spec['baseline_id'],'deduplicated':False,'hash':h}
 def _read_staged(self,path):
  parts=safe_relative(path).parts;fds=[]
  try:
   fd=os.open(self.root/'incoming',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);fds.append(fd)
   for part in parts[:-1]:fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);fds.append(fd)
   leaf=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd);fds.append(leaf);st=os.fstat(leaf);require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=MAX_ARTIFACT,'Invalid staged artifact')
   raw=b''
   while len(raw)<=MAX_ARTIFACT:
    chunk=os.read(leaf,min(1048576,MAX_ARTIFACT+1-len(raw)))
    if not chunk:break
    raw+=chunk
   require(len(raw)==st.st_size,'Artifact changed or too large');return raw
  finally:
   for fd in reversed(fds):os.close(fd)
 def _blob(self,body):
  path=self.root/'artifacts'/body['artifact_name'];require(not path.is_symlink() and path.is_file(),'Frozen artifact missing/symlink')
  require(digest(path.read_bytes())==body['source_sha256'],'Frozen source hash changed');return path
 def submit(self,spec):
  fields(spec,{'candidate_id','backend','baseline_id','entrypoint','artifact','artifact_kind','priority','submitted_by','contains_secrets'})
  require(identifier(spec['candidate_id']) and identifier(spec['baseline_id']) and identifier(spec['submitted_by']),'Invalid identifier')
  require(spec['backend'] in BACKENDS and spec['entrypoint'] in ENTRYPOINTS,'Invalid backend/entrypoint')
  require((spec['backend']=='CPU')==(spec['entrypoint']=='cpu_contract_v1'),'CPU contracts and GPU lanes must remain separate')
  require(type(spec['priority']) is int and 0<=spec['priority']<=100 and spec['contains_secrets'] is False,'Priority0..100 and no-secrets attestation required')
  raw=self._read_staged(spec['artifact']);inspection=validate_artifact(raw,spec['artifact_kind']);h=digest(raw);name=h+('.ipynb' if spec['artifact_kind']=='notebook' else '.zip');blob=self.root/'artifacts'/name


  fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=self.root/'artifacts')
  try:
   with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
   try:os.link(tmp,blob,follow_symlinks=False)
   except FileExistsError:require(not blob.is_symlink() and digest(blob.read_bytes())==h,'Artifact hash-path collision')
   dfd=os.open(self.root/'artifacts',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
   try:os.fsync(dfd)
   finally:os.close(dfd)
  finally:os.unlink(tmp)
  with self.tx() as c:
   baseline=c.execute('SELECT * FROM baselines WHERE id=?',(spec['baseline_id'],)).fetchone();require(baseline is not None and baseline['backend']==spec['backend'],'Matching registered backend/baseline required')
   b=json.loads(baseline['body']);body={k:spec[k] for k in ('candidate_id','backend','baseline_id','entrypoint','artifact_kind','submitted_by')};body.update(source_sha256=h,artifact_name=name,baseline_fingerprint=baseline['hash'],environment_fingerprint=b['environment_fingerprint'],task_fingerprint=b['task_fingerprint'],harness_commit=b['harness_commit'],inspection=inspection)
   key=digest(canonical({k:body[k] for k in ('candidate_id','backend','baseline_id','entrypoint','source_sha256')}).encode());old=c.execute('SELECT id FROM jobs WHERE dedup=?',(key,)).fetchone()
   if old:return {'job_id':old['id'],'deduplicated':True,'source_sha256':h}
   jid='q_'+uuid.uuid4().hex;c.execute('INSERT INTO jobs(id,dedup,body,priority,state,created) VALUES(?,?,?,?,?,?)',(jid,key,canonical(body),spec['priority'],'submitted',self.clock()));self.event(c,jid,'submitted',{'source_sha256':h,'priority':spec['priority']})
  return {'job_id':jid,'deduplicated':False,'source_sha256':h}
 def review(self,jid,*,reviewer,source_sha256,approved,authorization_ref):
  require(identifier(reviewer) and identifier(authorization_ref) and type(approved) is bool,'Explicit reviewer/authorization/boolean required')
  with self.tx() as c:
   j=c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone();require(j is not None,'Unknown job');b=json.loads(j['body']);require(source_sha256==b['source_sha256'],'Review source hash mismatch');self._blob(b)
   r={'reviewer':reviewer,'source_sha256':source_sha256,'approved':approved,'authorization_ref':authorization_ref};encoded=canonical(r)
   if j['review']==encoded:return {'job_id':jid,'state':j['state'],'idempotent':True}
   require(j['state']=='submitted','Review is immutable after decision');state='approved' if approved else 'rejected';c.execute('UPDATE jobs SET state=?,review=? WHERE id=?',(state,encoded,jid));self.event(c,jid,'review',r)
  return {'job_id':jid,'state':state,'idempotent':False}
 def _expire(self,c):
  rows=c.execute("SELECT * FROM attempts WHERE state IN ('claimed','running') AND lease_until<=?",(self.clock(),)).fetchall()
  for a in rows:
   c.execute("UPDATE attempts SET state='needs_reconciliation' WHERE id=?",(a['id'],));c.execute("UPDATE jobs SET state='needs_reconciliation' WHERE id=?",(a['job'],));self.event(c,a['job'],'lease_expired_no_rerun',{'attempt_id':a['id'],'resource':a['resource']})
  return len(rows)
 def expire_leases(self):
  with self.tx() as c:return {'quarantined':self._expire(c)}
 def claim(self,*,runner,backend,baseline_id,resource,lease_seconds=600):
  require(identifier(runner) and identifier(resource) and identifier(baseline_id) and backend in BACKENDS,'Invalid runner/resource/backend/baseline')
  require(type(lease_seconds) is int and 10<=lease_seconds<=3600,'Lease bound10..3600seconds')
  with self.tx() as c:
   self._expire(c)
   lock=c.execute('SELECT * FROM locks WHERE backend=? AND resource=?',(backend,resource)).fetchone()
   if lock:return {'claim':None,'reason':'resource_locked_including_uncertain_attempt','attempt_id':lock['attempt']}
   rows=c.execute("SELECT * FROM jobs WHERE state='approved' ORDER BY priority DESC,created ASC,id ASC").fetchall()
   j=next((j for j in rows if (lambda b:b['backend']==backend and b['baseline_id']==baseline_id)(json.loads(j['body']))),None)
   if j is None:return {'claim':None,'reason':'no_reviewed_matching_job'}
   b=json.loads(j['body']);path=self._blob(b);aid='a_'+uuid.uuid4().hex;now=self.clock()
   c.execute('INSERT INTO attempts(id,job,runner,backend,resource,state,claimed,lease_until) VALUES(?,?,?,?,?,?,?,?)',(aid,j['id'],runner,backend,resource,'claimed',now,now+lease_seconds));c.execute('INSERT INTO locks VALUES(?,?,?)',(backend,resource,aid));c.execute("UPDATE jobs SET state='claimed',attempt=? WHERE id=?",(aid,j['id']));self.event(c,j['id'],'claimed',{'attempt_id':aid,'runner':runner,'resource':resource,'lease_until':now+lease_seconds})
  return {'claim':{'job_id':j['id'],'attempt_id':aid,'runner':runner,'resource':resource,'lease_until':now+lease_seconds,'spec':b,'review':json.loads(j['review']),'reconciliation_history':[json.loads(e['body']) for e in self._events_for(j['id'],'manual_reconciliation')],'artifact_path':str(path),'contract':'Data only. Runner must independently verify authority, reviewed source, baseline, idle exclusive resource and unique output; no queue-provided shell command.'}}
 def _events_for(self,jid,kind):
  with contextlib.closing(self.connect()) as c:return c.execute('SELECT body FROM events WHERE job=? AND kind=? ORDER BY seq',(jid,kind)).fetchall()
 def _attempt(self,c,jid,aid,runner):
  a=c.execute('SELECT * FROM attempts WHERE id=? AND job=?',(aid,jid)).fetchone();require(a is not None and a['runner']==runner,'Attempt/runner mismatch');return a
 def heartbeat(self,jid,aid,runner,lease_seconds=600):
  require(type(lease_seconds) is int and 10<=lease_seconds<=3600,'Invalid lease bound')
  with self.tx() as c:
   a=self._attempt(c,jid,aid,runner);require(a['state'] in ('claimed','running') and a['lease_until']>self.clock(),'Expired/uncertain attempt requires explicit reconciliation');until=self.clock()+lease_seconds;c.execute('UPDATE attempts SET lease_until=? WHERE id=?',(until,aid));self.event(c,jid,'heartbeat',{'attempt_id':aid,'lease_until':until})
  return {'lease_until':until}
 def start(self,jid,aid,runner,run_reference):
  require(identifier(run_reference),'Opaque run reference required; no URL/command')
  with self.tx() as c:
   a=self._attempt(c,jid,aid,runner)
   if a['started'] is not None:require(a['run_reference']==run_reference,'Conflicting run identity');return {'state':a['state'],'idempotent':True}
   require(a['state']=='claimed' and a['lease_until']>self.clock(),'Cannot start expired/unclaimed work');c.execute("UPDATE attempts SET state='running',started=?,run_reference=? WHERE id=?",(self.clock(),run_reference,aid));c.execute("UPDATE jobs SET state='running' WHERE id=?",(jid,));self.event(c,jid,'start_confirmed',{'attempt_id':aid,'run_reference':run_reference})
  return {'state':'running','idempotent':False}
 def observe(self,jid,aid,runner,observation):
  fields(observation,{'kind','evidence_sha256','visible_step','expected_steps','route_evidence_state','route_counts'})
  require(observation['kind'] in {'stdout_progress','route_snapshot','artifact_observed'},'Unknown observation kind')
  require(sha(observation['evidence_sha256']),'Observation raw hash required; no raw secret/log payload')
  for k in ('visible_step','expected_steps'):require(observation[k] is None or type(observation[k]) is int and observation[k]>=0,'Invalid observed progress')
  require(observation['route_evidence_state'] in {'complete','partial','unknown'},'Invalid route evidence state')
  counts=observation['route_counts'];require(counts is None or type(counts) is dict and all(identifier(k) and type(v) is int and v>=0 for k,v in counts.items()),'Invalid route counts')
  require((counts is None)==(observation['route_evidence_state']=='unknown'),'Absent routes remain unknown/null')
  with self.tx() as c:
   a=self._attempt(c,jid,aid,runner);require(a['state'] in ('claimed','running','needs_reconciliation'),'Terminal attempt cannot append observations');self.event(c,jid,'observation',dict(observation,attempt_id=aid))
  return {'recorded':True,'candidate_status_unchanged':True}
 def mark_uncertain(self,jid,aid,runner,reason,evidence_sha256):
  require(reason in {'launch_outcome_unknown','observation_blocked','permission_denied','runner_crash_report'} and sha(evidence_sha256),'Explicit uncertainty reason/evidence required')
  with self.tx() as c:
   a=self._attempt(c,jid,aid,runner);require(a['state'] in ('claimed','running','needs_reconciliation'),'Terminal attempt cannot become uncertain');c.execute("UPDATE attempts SET state='needs_reconciliation' WHERE id=?",(aid,));c.execute("UPDATE jobs SET state='needs_reconciliation' WHERE id=?",(jid,));self.event(c,jid,'uncertain_no_retry',{'attempt_id':aid,'reason':reason,'evidence_sha256':evidence_sha256})
  return {'state':'needs_reconciliation','resource_lock_retained':True}
 def reconcile(self,jid,aid,*,reviewer,evidence_sha256,decision,authorization_ref):
  require(identifier(reviewer) and sha(evidence_sha256) and identifier(authorization_ref),'Explicit reconciliation evidence and renewed authority required')
  require(decision=='requeue_confirmed_not_started','Only confirmed-not-started requeue supported; uncertain/running work stays locked until terminal result')
  with self.tx() as c:
   self._expire(c);j=c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone();require(j is not None and authorization_ref!=json.loads(j['review'])['authorization_ref'],'A distinct renewed authorization reference is required');a=c.execute('SELECT * FROM attempts WHERE id=? AND job=?',(aid,jid)).fetchone();require(a is not None and a['state']=='needs_reconciliation' and a['started'] is None,'Only stale never-confirmed-started attempts qualify')
   c.execute("UPDATE attempts SET state='requeued_confirmed_not_started' WHERE id=?",(aid,));c.execute('DELETE FROM locks WHERE attempt=?',(aid,));c.execute("UPDATE jobs SET state='approved',attempt=NULL WHERE id=?",(jid,));self.event(c,jid,'manual_reconciliation',{'attempt_id':aid,'decision':decision,'reviewer':reviewer,'authorization_ref':authorization_ref,'evidence_sha256':evidence_sha256,'warning':'Reviewer must check external execution; absence of start record alone is NOT proof.'})
  return {'state':'approved','previous_attempt_preserved':True}
 def _validate_result(self,result,b):
  fields(result,{'schema_version','candidate_id','source_sha256','backend','baseline_id','environment_fingerprint','task_fingerprint','harness_commit','outcome','execution','metrics','provenance','preflight'})
  require(result['schema_version']==1,'Unknown result schema')
  for k in ('candidate_id','source_sha256','backend','baseline_id','environment_fingerprint','task_fingerprint','harness_commit'):require(result[k]==b[k],'Result identity/baseline mismatch: '+k)
  require(result['outcome'] in {'completed','aborted','failed','preflight_failed'},'Invalid terminal outcome')
  e=result['execution'];fields(e,{'run_reference','command_exit_code','wrapper_exit_code','nested_metrics_exit_code','verified_global_step','visible_step','expected_steps','adapter_saved','wrapper_elapsed_s','termination_reason','task_completion_status'})
  require(identifier(e['run_reference']),'Invalid run reference')
  require(e['termination_reason'] in {'completed','timeout','signal','command_error','preflight_error','task_incomplete','unknown'},'Explicit termination reason required')
  require(e['task_completion_status'] in {'VERIFIED_COMPLETE','INCOMPLETE','UNKNOWN','NOT_APPLICABLE'},'Explicit task-completion evidence state required')
  for k in ('command_exit_code','wrapper_exit_code','nested_metrics_exit_code','verified_global_step','visible_step','expected_steps'):require(e[k] is None or type(e[k]) is int,'Exit/step values must be integer or null')
  require(all(e[k] is None or e[k]>=0 for k in ('verified_global_step','visible_step','expected_steps')),'Negative progress')
  require(e['adapter_saved'] is None or type(e['adapter_saved']) is bool,'Adapter saved is true/false/null, not inferred from steps')
  require(e['wrapper_elapsed_s'] is None or finite_number(e['wrapper_elapsed_s']),'Invalid wrapper elapsed duration')
  pre=result['preflight'];fields(pre,{'status','scope','raw_sha256','source_sha256'})
  require(pre['status'] in {'PASS','FAIL','NOT_RUN','UNKNOWN'} and pre['scope'] in {'CPU_CONTRACT','T4_PREFLIGHT','GCP_PREFLIGHT','NOT_RUN','UNKNOWN'},'Invalid separate preflight scope/status')
  require(pre['source_sha256'] is None or pre['source_sha256']==b['source_sha256'],'Preflight source mismatch')
  require(pre['raw_sha256'] is None or sha(pre['raw_sha256']),'Invalid preflight evidence hash')
  if pre['status'] in {'PASS','FAIL'}:require(sha(pre['raw_sha256']) and pre['source_sha256']==b['source_sha256'] and pre['scope'] not in {'NOT_RUN','UNKNOWN'},'Preflight decision requires scoped same-source raw evidence')
  m=result['metrics'];fields(m,{'command_runtime_s','observed_runtime_s','runtime_kind','peak_memory_mib','peak_kind','memory_scope','eval_loss','own_kernel_share','admission','score','runtime_scope'})
  require(m['runtime_scope']=={'cpu_contract_v1':'cpu_contract_process','gpu_preflight_v1':'gpu_preflight_process','soup_paired_v1':'paired_command_median','soup_screen_v1':'native_train_command'}[b['entrypoint']],'Runtime scope must match reviewed entrypoint')
  for k in ('command_runtime_s','observed_runtime_s','peak_memory_mib','eval_loss','own_kernel_share','score'):require(m[k] is None or finite_number(m[k]),'Metric must be finite nonnegative or null: '+k)
  require(m['runtime_kind'] in {'complete','right_censored','error_duration','partial_observation','unavailable'} and m['peak_kind'] in {'complete','partial','unavailable'},'Invalid censoring flags')
  require(m['memory_scope'] in {'gpu_physical_sum','cpu_rss','unknown'},'Explicit memory scope required')
  require(m['own_kernel_share'] is None or m['own_kernel_share']<=1,'Own share is a fraction0..1, not percent')
  require(m['admission'] in {'PASS','FAIL','UNKNOWN'},'Invalid admission status')
  if result['outcome']=='completed':
   require(e['termination_reason']=='completed','Completed outcome requires completed termination')
   if b['entrypoint'] in {'soup_paired_v1','soup_screen_v1'}:require(e['task_completion_status']=='VERIFIED_COMPLETE' and e['adapter_saved'] is True and e['expected_steps'] is not None and e['expected_steps']>0 and e['verified_global_step']==e['expected_steps'],'Task success needs verified steps and saved artifact, not wrapper/progress alone')
   require(e['command_exit_code']==0 and e['wrapper_exit_code'] in (None,0) and e['nested_metrics_exit_code'] in (None,0),'Outer or nested failure cannot be completed; command exit takes precedence')
   require(m['runtime_kind']=='complete' and m['command_runtime_s'] is not None,'Completed runtime required')
   require(m['observed_runtime_s'] in (None,m['command_runtime_s']),'Observed complete runtime must match')
  else:
   require(m['command_runtime_s'] is None and m['eval_loss'] is None and m['own_kernel_share'] is None and m['score'] is None and m['admission']=='UNKNOWN','Partial/failed run cannot receive complete time/eval/share/score/admission')
   require(e['termination_reason']!='completed','Noncompleted outcome needs termination reason')
   if result['outcome']=='aborted':require(e['termination_reason'] in {'timeout','signal','unknown'},'Aborted outcome requires interruption/unknown reason')
   expected_runtime='unavailable' if m['observed_runtime_s'] is None else 'right_censored' if e['termination_reason'] in {'timeout','signal'} else 'partial_observation' if e['termination_reason']=='unknown' else 'error_duration'
   require(m['runtime_kind']==expected_runtime,'Duration classification must distinguish interruption censoring from command/preflight error duration')
   require(m['peak_kind'] in ('partial','unavailable'),'Incomplete peak cannot be complete')
  require((m['peak_memory_mib'] is None)==(m['peak_kind']=='unavailable'),'Peak value/completeness mismatch')
  if b['entrypoint'] in {'cpu_contract_v1','gpu_preflight_v1'}:require(m['eval_loss'] is None and m['own_kernel_share'] is None and m['admission']=='UNKNOWN' and m['score'] is None,'Contract/preflight metrics are not task evaluation/share/admission')
  if b['backend']=='CPU':require(m['memory_scope'] in ('cpu_rss','unknown') and m['own_kernel_share'] is None,'CPU is not GPU memory/kernel-share evidence')
  p=result['provenance'];fields(p,{'raw_result_sha256','checks_sha256','collector','admission_reviewer','score_evidence_sha256','runtime_json_pointer','peak_json_pointer','actual_route_counts','own_kernel_names','executed_source_manifest_sha256','source_set_matches_reviewed','route_evidence_state','source_allowlist_sha256','route_evidence_scope','activation_status','own_share_scope','own_share_evidence_sha256'})
  require(sha(p['raw_result_sha256']) and (p['checks_sha256'] is None or sha(p['checks_sha256'])) and identifier(p['collector']),'Raw result/collector provenance required')
  for k in ('runtime_json_pointer','peak_json_pointer'):require(p[k] is None or isinstance(p[k],str) and p[k].startswith('/') and len(p[k])<=256,'Invalid provenance JSON pointer')
  require(p['actual_route_counts'] is None or type(p['actual_route_counts']) is dict and all(identifier(k) and type(v) is int and v>=0 for k,v in p['actual_route_counts'].items()),'Invalid actual route counts; absent evidence must be null')
  require(p['own_kernel_names'] is None or type(p['own_kernel_names']) is list and all(identifier(k) for k in p['own_kernel_names']),'Invalid own-kernel names; absent evidence must be null')
  require(p['route_evidence_state'] in {'complete','partial','unknown'},'Route evidence state required')
  require(p['route_evidence_scope'] in {'task','preflight','cpu_contract','unknown'} and p['activation_status'] in {'OBSERVED_USEFUL','NATIVE_FALLBACK_ONLY','GUARD_ONLY','NOT_OBSERVED','UNKNOWN'},'Route scope/useful activation evidence required; installer success is not activation')
  if p['route_evidence_state']=='unknown':require(p['route_evidence_scope']=='unknown' and p['activation_status']=='UNKNOWN','Missing route evidence remains unknown in every route field')
  else:require(p['route_evidence_scope']!='unknown','Known route evidence needs stage scope')
  require((p['actual_route_counts'] is None)==(p['route_evidence_state']=='unknown'),'Missing route evidence is null/unknown, never a zero count')
  require(p['source_allowlist_sha256'] is None or sha(p['source_allowlist_sha256']),'Invalid own+pristine-support allowlist hash')
  require(p['executed_source_manifest_sha256'] is None or sha(p['executed_source_manifest_sha256']),'Invalid executed source manifest hash')
  require(p['source_set_matches_reviewed'] is None or type(p['source_set_matches_reviewed']) is bool,'Source set match is true/false/null')
  require(p['source_set_matches_reviewed'] is None or sha(p['executed_source_manifest_sha256']) and sha(p['source_allowlist_sha256']),'Source set comparison requires manifest and reviewed own+explicit pinned-support allowlist hashes')
  require(p['own_share_scope'] in {'task','unknown'},'Task own-share field cannot inherit CPU/preflight scope')
  if m['own_kernel_share'] is not None:require(p['own_share_scope']=='task' and sha(p['own_share_evidence_sha256']) and p['own_kernel_names'] is not None and (m['own_kernel_share']==0 or bool(p['own_kernel_names'])),'Own share requires task-scoped raw timing evidence and nonempty names for positive share, independently of optional route telemetry')
  else:require(p['own_share_scope']=='unknown' and p['own_share_evidence_sha256'] is None,'Missing own share remains null/unknown')
  if m['admission']=='PASS':require(m['own_kernel_share'] is not None and m['eval_loss'] is not None,'Admission PASS requires task share and evaluation evidence')
  if m['admission']!='UNKNOWN':require(b['backend']=='KAGGLE_T4' and b['entrypoint']=='soup_paired_v1' and sha(p['checks_sha256']) and identifier(p['admission_reviewer']) and p['source_set_matches_reviewed'] is True,'Admission requires reviewed paired Kaggle evidence, not CPU/GCP/preflight')
  else:require(p['admission_reviewer'] is None,'Unknown admission cannot name an approving reviewer')
  if m['score'] is not None:require(m['admission']=='PASS' and sha(p['score_evidence_sha256']),'Score requires separate external scoring evidence; queue never computes a score')
  else:require(p['score_evidence_sha256'] is None,'No score means no score provenance')
 def complete(self,jid,aid,runner,result):
  encoded=canonical(result);rh=digest(encoded.encode())
  with self.tx() as c:
   a=self._attempt(c,jid,aid,runner);j=c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone();b=json.loads(j['body']);self._validate_result(result,b)
   if j['result'] is not None:require(j['attempt']==aid and j['result']==encoded,'Completion conflict: immutable result already exists');return {'state':j['state'],'result_sha256':j['result_hash'],'idempotent':True}
   require(j['attempt']==aid and a['state'] in ('claimed','running','needs_reconciliation'),'Attempt is not current/claimable')
   if a['run_reference'] is not None:require(result['execution']['run_reference']==a['run_reference'],'External run identity changed')
   self._blob(b);state=result['outcome'];c.execute('UPDATE jobs SET state=?,result=?,result_hash=? WHERE id=?',(state,encoded,rh,jid));c.execute('UPDATE attempts SET state=? WHERE id=?',(state,aid));c.execute('DELETE FROM locks WHERE attempt=?',(aid,));self.event(c,jid,'completed_immutable',{'attempt_id':aid,'outcome':state,'result_sha256':rh})
  return {'state':state,'result_sha256':rh,'idempotent':False}
 def status(self,jid=None):
  with contextlib.closing(self.connect()) as c:
   rows=c.execute('SELECT * FROM jobs'+(' WHERE id=?' if jid else '')+' ORDER BY created,id',((jid,) if jid else ())).fetchall();out=[]
   for j in rows:
    d=dict(j)
    for k in ('body','review','result'):d[k]=json.loads(d[k]) if d[k] is not None else None
    d['attempts']=[dict(a) for a in c.execute('SELECT * FROM attempts WHERE job=? ORDER BY claimed,id',(j['id'],))];d['events']=[dict(e)|{'body':json.loads(e['body'])} for e in c.execute('SELECT * FROM events WHERE job=? ORDER BY seq',(j['id'],))];out.append(d)
   if jid:require(len(out)==1,'Unknown job');return out[0]
   return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);s=p.add_subparsers(dest='op',required=True)
 for op in ('baseline','submit'):
  x=s.add_parser(op);x.add_argument('--spec',required=True)
 x=s.add_parser('status');x.add_argument('--job')
 x=s.add_parser('review');x.add_argument('--job',required=True);x.add_argument('--reviewer',required=True);x.add_argument('--source-sha256',required=True);x.add_argument('--authorization-ref',required=True);x.add_argument('--decision',choices=['approve','reject'],required=True)
 x=s.add_parser('claim');x.add_argument('--runner',required=True);x.add_argument('--backend',choices=sorted(BACKENDS),required=True);x.add_argument('--baseline-id',required=True);x.add_argument('--resource',required=True);x.add_argument('--lease-seconds',type=int,default=600)
 for op in ('start','heartbeat','complete','observe','uncertain'):
  x=s.add_parser(op);x.add_argument('--job',required=True);x.add_argument('--attempt',required=True);x.add_argument('--runner',required=True)
  if op=='start':x.add_argument('--run-reference',required=True)
  if op=='heartbeat':x.add_argument('--lease-seconds',type=int,default=600)
  if op=='complete':x.add_argument('--result',required=True)
  if op=='observe':x.add_argument('--observation',required=True)
  if op=='uncertain':x.add_argument('--reason',required=True);x.add_argument('--evidence-sha256',required=True)
 s.add_parser('expire')
 x=s.add_parser('reconcile');x.add_argument('--job',required=True);x.add_argument('--attempt',required=True);x.add_argument('--reviewer',required=True);x.add_argument('--evidence-sha256',required=True);x.add_argument('--decision',choices=['requeue_confirmed_not_started'],required=True);x.add_argument('--authorization-ref',required=True)
 a=p.parse_args();q=Queue(a.root)
 if a.op=='baseline':r=q.register_baseline(json.loads(Path(a.spec).read_text()))
 elif a.op=='submit':r=q.submit(json.loads(Path(a.spec).read_text()))
 elif a.op=='status':r=q.status(a.job)
 elif a.op=='review':r=q.review(a.job,reviewer=a.reviewer,source_sha256=a.source_sha256,approved=a.decision=='approve',authorization_ref=a.authorization_ref)
 elif a.op=='claim':r=q.claim(runner=a.runner,backend=a.backend,baseline_id=a.baseline_id,resource=a.resource,lease_seconds=a.lease_seconds)
 elif a.op=='start':r=q.start(a.job,a.attempt,a.runner,a.run_reference)
 elif a.op=='heartbeat':r=q.heartbeat(a.job,a.attempt,a.runner,a.lease_seconds)
 elif a.op=='complete':r=q.complete(a.job,a.attempt,a.runner,json.loads(Path(a.result).read_text()))
 elif a.op=='observe':r=q.observe(a.job,a.attempt,a.runner,json.loads(Path(a.observation).read_text()))
 elif a.op=='uncertain':r=q.mark_uncertain(a.job,a.attempt,a.runner,a.reason,a.evidence_sha256)
 elif a.op=='expire':r=q.expire_leases()
 elif a.op=='reconcile':r=q.reconcile(a.job,a.attempt,reviewer=a.reviewer,evidence_sha256=a.evidence_sha256,decision=a.decision,authorization_ref=a.authorization_ref)
 print(json.dumps(r,indent=2,allow_nan=False))
if __name__=='__main__':
 try:main()
 except (QueueError,OSError,sqlite3.Error,json.JSONDecodeError,TypeError) as e:print(json.dumps({'error':type(e).__name__,'message':str(e)}));raise SystemExit(2)
