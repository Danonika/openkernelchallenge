#!/usr/bin/env python3
"""Small header-auth queue client. Never executes submissions; no automatic retries."""
import argparse,base64,hashlib,json,os,re,sys,urllib.error,urllib.parse,urllib.request
from pathlib import Path
MAX_RESPONSE=16*1024*1024
class ClientError(RuntimeError):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise ClientError('Redirect refused; credentials stay on the configured endpoint')
class Client:
 def __init__(self,base_url,token,timeout=20):
  p=urllib.parse.urlsplit(base_url)
  if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.path not in ('','/') or p.query or p.fragment:raise ClientError('Use a canonical HTTP(S) origin without credentials/path/query')
  if p.scheme=='http' and p.hostname not in ('127.0.0.1','localhost','::1'):raise ClientError('Plain HTTP is allowed only on loopback/SSH tunnel; remote endpoints require HTTPS')
  if not isinstance(token,str) or re.fullmatch(r'[A-Za-z0-9_-]{32,256}',token) is None:raise ClientError('Valid scoped token must come from the selected environment variable')
  self.base=base_url.rstrip('/');self.token=token;self.timeout=timeout
  self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
 def _request(self,method,path,body=None):
  if not re.fullmatch(r'/[A-Za-z0-9_/-]+',path) or '..' in path:raise ClientError('Noncanonical API path')
  data=None if body is None else json.dumps(body,allow_nan=False).encode()
  headers={'Authorization':'Bearer '+self.token,'Accept':'application/json'}
  if data is not None:headers['Content-Type']='application/json'
  req=urllib.request.Request(self.base+path,data=data,headers=headers,method=method)
  try:
   with self.opener.open(req,timeout=self.timeout) as response:
    raw=response.read(MAX_RESPONSE+1)
    if len(raw)>MAX_RESPONSE:raise ClientError('Response exceeds client bound')
    return raw,dict(response.headers)
  except urllib.error.HTTPError as e:
   # Do not print request/header objects or arbitrary proxy error bodies.
   raise ClientError('Queue HTTP '+str(e.code)+'; no automatic retry') from None
  except urllib.error.URLError:raise ClientError('Queue unavailable/connection uncertain; inspect status before retrying actions') from None
 def request(self,method,path,body=None):
  raw,_=self._request(method,path,body)
  try:return json.loads(raw)
  except (ValueError,UnicodeDecodeError):raise ClientError('Expected queue JSON response') from None
 def download(self,job_id,destination,expected_sha256):
  if not re.fullmatch(r'q_[0-9a-f]{32}',job_id) or not re.fullmatch(r'[0-9a-f]{64}',expected_sha256):raise ClientError('Invalid job/hash')
  raw,headers=self._request('GET','/v1/jobs/'+job_id+'/artifact')
  if len(raw)>8*1024*1024 or hashlib.sha256(raw).hexdigest()!=expected_sha256 or headers.get('X-Artifact-SHA256')!=expected_sha256:raise ClientError('Downloaded source hash/bounds mismatch')
  path=Path(destination)
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  return {'path':str(path),'sha256':expected_sha256,'bytes':len(raw)}
def read_json(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8765');p.add_argument('--token-env',default='SOUP_QUEUE_AGENT_TOKEN');s=p.add_subparsers(dest='action',required=True)
 s.add_parser('session');x=s.add_parser('status');x.add_argument('--job')
 x=s.add_parser('baseline');x.add_argument('--spec',required=True)
 x=s.add_parser('submit');x.add_argument('--spec',required=True);x.add_argument('--artifact',required=True)
 x=s.add_parser('review');x.add_argument('--job',required=True);x.add_argument('--sha256',required=True);x.add_argument('--authorization-ref',required=True);x.add_argument('--decision',choices=['approve','reject'],required=True)
 x=s.add_parser('claim');x.add_argument('--backend',required=True);x.add_argument('--baseline-id',required=True);x.add_argument('--resource',required=True);x.add_argument('--lease-seconds',type=int,default=600)
 x=s.add_parser('download');x.add_argument('--job',required=True);x.add_argument('--sha256',required=True);x.add_argument('--out',required=True)
 for name in ('start','heartbeat','observe','uncertain','complete','reconcile'):
  x=s.add_parser(name);x.add_argument('--job',required=True);x.add_argument('--attempt',required=True);x.add_argument('--body',required=True)
 a=p.parse_args();c=Client(a.url,os.environ.get(a.token_env,''))
 if a.action=='session':r=c.request('GET','/v1/session')
 elif a.action=='status':r=c.request('GET','/v1/jobs'+('/'+a.job if a.job else ''))
 elif a.action=='baseline':r=c.request('POST','/v1/baselines',read_json(a.spec))
 elif a.action=='submit':
  raw=Path(a.artifact).read_bytes()
  if not 0<len(raw)<=8*1024*1024:raise ClientError('Artifact must be1..8MiB')
  body=read_json(a.spec);body['artifact_base64']=base64.b64encode(raw).decode();r=c.request('POST','/v1/jobs',body)
 elif a.action=='review':r=c.request('POST',f'/v1/jobs/{a.job}/review',{'source_sha256':a.sha256,'approved':a.decision=='approve','authorization_ref':a.authorization_ref})
 elif a.action=='claim':r=c.request('POST','/v1/claims',{'backend':a.backend,'baseline_id':a.baseline_id,'resource':a.resource,'lease_seconds':a.lease_seconds})
 elif a.action=='download':r=c.download(a.job,a.out,a.sha256)
 else:r=c.request('POST',f'/v1/jobs/{a.job}/attempts/{a.attempt}/{a.action}',read_json(a.body))
 print(json.dumps(r,indent=2,allow_nan=False))
if __name__=='__main__':
 try:main()
 except (ClientError,OSError,ValueError) as e:print(json.dumps({'error':type(e).__name__,'message':str(e)}),file=sys.stderr);raise SystemExit(2)
