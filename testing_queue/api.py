#!/usr/bin/env python3
"""Loopback-only authenticated API prototype. No artifact execution or deployment."""
import argparse,base64,binascii,hmac,json,os,re,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from soup_queue import Queue,QueueError,MAX_ARTIFACT,fields,require
import notebook_queue
MAX_BODY=12*1024*1024
ROLE_SCOPES={'submitter':{'read','submit'},'reviewer':{'read','review','artifact'},'worker':{'read','work','artifact'},'viewer':{'read'}}
ROOT=Path(__file__).resolve().parent

def credential_table(tokens):
 require(set(tokens)<=set(ROLE_SCOPES) and {'submitter','reviewer','worker'}<=set(tokens),'Explicit separate submitter/reviewer/worker credentials required')
 seen=set();out=[]
 for role,token in tokens.items():
  require(isinstance(token,str) and re.fullmatch(r'[A-Za-z0-9_-]{32,256}',token),'Bearer credentials must be strong externally provided tokens, not short passwords')
  require(token not in seen,'Role tokens must be distinct');seen.add(token);out.append((token,'api-'+role,ROLE_SCOPES[role],role))
 return out

class LocalServer(ThreadingHTTPServer):
 daemon_threads=True;allow_reuse_address=True
 def __init__(self,queue_root,tokens,port=0,trusted_public_origin=None,notebook_config=None):
  self.credentials=credential_table(tokens)
  if trusted_public_origin is not None:
   origin=urlsplit(trusted_public_origin);require(origin.scheme=='https' and bool(origin.netloc) and origin.username is None and origin.password is None and not origin.path and not origin.query and not origin.fragment,'Only an explicit canonical HTTPS origin may be trusted for future reverse proxy')
  self.queue=Queue(queue_root)
  self.notebooks=notebook_queue.NotebookQueue(queue_root,notebook_config) if notebook_config else None
  super().__init__(('127.0.0.1',port),Handler)
  self.allowed_hosts={f'127.0.0.1:{self.server_port}',f'localhost:{self.server_port}'}
  self.allowed_origins={'http://'+x for x in self.allowed_hosts}
  if trusted_public_origin is not None:
   self.allowed_origins.add(trusted_public_origin)

class Handler(BaseHTTPRequestHandler):
 server_version='SoupQueueLocal/1';sys_version=''
 def setup(self):super().setup();self.connection.settimeout(10)
 def log_message(self,*args):pass  # Never log URLs, bodies, Authorization or credentials.
 def _send(self,status,body,content_type='application/json',extra=None):
  data=body if isinstance(body,bytes) else json.dumps(body,allow_nan=False).encode()
  self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.send_header('X-Frame-Options','DENY');self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
  for k,v in (extra or {}).items():self.send_header(k,v)
  self.end_headers();self.wfile.write(data)
 def _route(self):
  # No URL/query passwords, callbacks, redirects, CORS or alternate bind policy.
  require(self.headers.get('Host') in self.server.allowed_hosts,'Unexpected Host; local prototype only')
  parsed=urlsplit(self.path);require(not parsed.query and not parsed.fragment and not parsed.scheme and not parsed.netloc,'Query strings/URL credentials are forbidden; use Authorization header')
  origin=self.headers.get('Origin');require(origin is None or origin in self.server.allowed_origins,'Cross-origin access rejected')
  return parsed.path
 def _auth(self,scope):
  authorization=self.headers.get('Authorization','')
  if len(self.headers.get_all('Authorization',[]))!=1:self._send(401,{'error':'One Authorization header required'});return None
  if not authorization.startswith('Bearer '):self._send(401,{'error':'Authorization: Bearer header required'});return None
  supplied=authorization[7:];matched=None
  if re.fullmatch(r'[A-Za-z0-9_-]{32,256}',supplied):
   for token,subject,scopes,role in self.server.credentials:
    if hmac.compare_digest(supplied,token):matched=(subject,scopes,role)
  if matched is None:self._send(401,{'error':'Invalid credentials'});return None
  if scope not in matched[1]:self._send(403,{'error':'Insufficient scope'});return None
  return matched
 def _json(self):
  require(self.headers.get('Transfer-Encoding') is None,'Transfer encoding not supported')
  require(self.headers.get('Content-Type','').split(';',1)[0]=='application/json','application/json required')
  try:length=int(self.headers.get('Content-Length',''))
  except ValueError:raise QueueError('Content-Length required')
  if not 0<length<=MAX_BODY:self._send(413,{'error':'Request body exceeds fixed limit or is empty'});return None
  raw=self.rfile.read(length);require(len(raw)==length,'Truncated body')
  def unique(pairs):
   d={}
   for k,v in pairs:require(k not in d,'Duplicate JSON key');d[k]=v
   return d
  value=json.loads(raw,object_pairs_hook=unique);require(type(value) is dict,'Request JSON object required');return value
 def do_GET(self):
  try:
   path=self._route()
   static={'/':('dashboard.html','text/html; charset=utf-8'),'/dashboard.js':('dashboard.js','text/javascript; charset=utf-8'),'/dashboard.css':('dashboard.css','text/css; charset=utf-8'),'/notebooks':('notebooks.html','text/html; charset=utf-8'),'/notebooks.js':('notebooks.js','text/javascript; charset=utf-8')}
   if self.server.notebooks:
    static['/']=('notebooks.html','text/html; charset=utf-8')
    static['/bundles']=('dashboard.html','text/html; charset=utf-8')
   static['/notebooks.css']=('notebooks.css','text/css; charset=utf-8')
   if path in static:
    name,mime=static[path];self._send(200,(ROOT/'web'/name).read_bytes(),mime);return
   downloads={'/notebook_client.py':'notebook_client.py','/notebook-gpu-smoke.ipynb':'examples/notebook-gpu-smoke.ipynb','/notebook-training-template.ipynb':'examples/notebook-training-template.ipynb','/notebook-storage-probe.ipynb':'examples/notebook-storage-probe.ipynb'}
   if self.server.notebooks and path in downloads:
    self._send(200,(ROOT/downloads[path]).read_bytes(),'application/octet-stream',{'Content-Disposition':'attachment; filename="'+Path(downloads[path]).name+'"'});return
   if notebook_queue.handle_get(self,path):return
   if path=='/v1/session':
    actor=self._auth('read')
    if actor:self._send(200,{'subject':actor[0],'role':actor[2],'scopes':sorted(actor[1]),'mode':'notebook-service' if self.server.notebooks else 'loopback-prototype'})
    return
   if path=='/v1/jobs':
    if self._auth('read'):self._send(200,{'jobs':self.server.queue.status()})
    return
   m=re.fullmatch(r'/v1/jobs/(q_[0-9a-f]{32})(/artifact)?',path)
   if m:
    if not self._auth('artifact' if m[2] else 'read'):return
    job=self.server.queue.status(m[1])
    if m[2]:
     artifact=self.server.queue._blob(job['body']);self._send(200,artifact.read_bytes(),'application/octet-stream',{'Content-Disposition':'attachment; filename="'+job['body']['artifact_name']+'"','X-Artifact-SHA256':job['body']['source_sha256']})
    else:self._send(200,job)
    return
   self._send(404,{'error':'Unknown route'})
  except (QueueError,ValueError,OSError,json.JSONDecodeError) as e:self._send(400,{'error':type(e).__name__,'message':str(e)})
 def do_POST(self):
  try:
   path=self._route();q=self.server.queue
   if notebook_queue.handle_post(self,path):return
   if path=='/v1/jobs':scope='submit'
   elif path=='/v1/baselines' or path.endswith('/review') or path.endswith('/reconcile'):scope='review'
   else:scope='work'
   actor=self._auth(scope)
   if not actor:return
   body=self._json()
   if body is None:return
   subject=actor[0]
   if path=='/v1/jobs':
    fields(body,{'candidate_id','backend','baseline_id','entrypoint','artifact_kind','priority','contains_secrets','artifact_base64'})
    require(isinstance(body['artifact_base64'],str) and len(body['artifact_base64'])<=4*((MAX_ARTIFACT+2)//3),'Upload too large')
    try:raw=base64.b64decode(body['artifact_base64'],validate=True)
    except (ValueError,binascii.Error) as e:raise QueueError('Invalid base64 artifact') from e
    require(0<len(raw)<=MAX_ARTIFACT,'Upload bounds exceeded')
    name='http-'+uuid.uuid4().hex+('.ipynb' if body['artifact_kind']=='notebook' else '.zip');path_on_disk=q.root/'incoming'/name
    try:
     fd=os.open(path_on_disk,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
     with os.fdopen(fd,'wb') as f:f.write(raw)
     spec={k:v for k,v in body.items() if k!='artifact_base64'};spec.update(artifact=name,submitted_by=subject);result=q.submit(spec)
    finally:path_on_disk.unlink(missing_ok=True)
    self._send(201,result);return
   if path=='/v1/baselines':self._send(200,q.register_baseline(body));return
   if path=='/v1/claims':
    fields(body,{'backend','baseline_id','resource'},{'lease_seconds'});self._send(200,q.claim(runner=subject,**body));return
   if path=='/v1/leases/expire':fields(body,set());self._send(200,q.expire_leases());return
   review=re.fullmatch(r'/v1/jobs/(q_[0-9a-f]{32})/review',path)
   if review:
    fields(body,{'source_sha256','approved','authorization_ref'});self._send(200,q.review(review[1],reviewer=subject,**body));return
   attempt=re.fullmatch(r'/v1/jobs/(q_[0-9a-f]{32})/attempts/(a_[0-9a-f]{32})/(start|heartbeat|observe|uncertain|complete|reconcile)',path)
   if attempt:
    jid,aid,action=attempt.groups()
    if action=='start':fields(body,{'run_reference'});result=q.start(jid,aid,subject,body['run_reference'])
    elif action=='heartbeat':fields(body,set(),{'lease_seconds'});result=q.heartbeat(jid,aid,subject,body.get('lease_seconds',600))
    elif action=='observe':result=q.observe(jid,aid,subject,body)
    elif action=='uncertain':fields(body,{'reason','evidence_sha256'});result=q.mark_uncertain(jid,aid,subject,body['reason'],body['evidence_sha256'])
    elif action=='complete':result=q.complete(jid,aid,subject,body)
    else:fields(body,{'evidence_sha256','decision','authorization_ref'});result=q.reconcile(jid,aid,reviewer=subject,**body)
    self._send(200,result);return
   self._send(404,{'error':'Unknown route'})
  except (QueueError,ValueError,OSError,TypeError,json.JSONDecodeError) as e:self._send(400,{'error':type(e).__name__,'message':str(e)})
 def do_OPTIONS(self):self._send(403,{'error':'Cross-origin access is not enabled'})

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--serve-local',action='store_true',required=True);p.add_argument('--trusted-public-origin');p.add_argument('--notebook-config');a=p.parse_args()
 tokens={role:os.environ['SOUP_QUEUE_'+role.upper()+'_TOKEN'] for role in ('submitter','reviewer','worker')}
 if os.environ.get('SOUP_QUEUE_VIEWER_TOKEN'):tokens['viewer']=os.environ['SOUP_QUEUE_VIEWER_TOKEN']
 server=LocalServer(a.root,tokens,a.port,a.trusted_public_origin,json.loads(Path(a.notebook_config).read_text()) if a.notebook_config else None)
 print(json.dumps({'status':'ready','bind':'127.0.0.1','port':server.server_port,'trusted_public_origin':a.trusted_public_origin,'automatic_notebooks':bool(a.notebook_config)}),flush=True)
 try:server.serve_forever()
 finally:server.server_close()
if __name__=='__main__':main()
