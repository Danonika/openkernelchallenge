#!/usr/bin/env python3
"""USER-RUN ONLY: write new scoped credentials, never print them or overwrite files."""
import argparse,os,secrets
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--confirm-create-secrets',action='store_true',required=True);a=p.parse_args()
root=Path(a.directory).absolute()
if root.exists():raise SystemExit('Choose a NEW secret directory; no overwrite or deletion is performed')
root.mkdir(mode=0o700,parents=False)
tokens={role:secrets.token_hex(32) for role in ('submitter','reviewer','worker','viewer')}
outputs={'api.env':''.join('SOUP_QUEUE_'+r.upper()+'_TOKEN='+t+'\n' for r,t in tokens.items())}
outputs.update({r+'.env':('SOUP_QUEUE_WORKER_TOKEN' if r=='worker' else 'SOUP_QUEUE_AGENT_TOKEN')+'='+t+'\n' for r,t in tokens.items()})
for name,value in outputs.items():
 fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
print('Created separate credential files with mode0600. Values were not printed. Distribute only the intended role file over an approved secure channel.')
