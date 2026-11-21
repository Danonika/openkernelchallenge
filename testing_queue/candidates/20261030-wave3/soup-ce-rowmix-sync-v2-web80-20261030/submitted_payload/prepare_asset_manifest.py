#!/usr/bin/env python3
"""Read/hash only the selected public model cache + public task files; no download."""
import argparse,hashlib,json,re
from pathlib import Path
MODEL='models--Qwen--Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
DATA={'data/train.jsonl':'93d036d912bd81c337dc29893e13a76f36161d96f896ef4e02492e2f1a21a0dd','data/eval_public.jsonl':'b7f85b78ba3186dce4deab7cfb8bf8dcf7998e1517c5749ac0f5005886c6f70e'}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def build(cache,repo,revision=REVISION):
 cache=Path(cache).resolve();repo=Path(repo).resolve()
 if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Revision must be40hex')
 root=cache/MODEL
 if root.is_symlink() or not root.is_dir():raise ValueError('Selected model cache missing or symlinked')
 for directory in (root/'refs',root/'snapshots',root/'blobs'):
  if directory.is_symlink() or not directory.is_dir():raise ValueError('Model metadata/blob directories must be real local directories')
 ref=root/'refs/main'
 if ref.is_symlink() or not ref.is_file() or ref.read_text().strip()!=revision:raise ValueError('refs/main must identify exact pinned revision; no silent revision substitution')
 snapshot=root/'snapshots'/revision
 if snapshot.is_symlink() or not snapshot.is_dir():raise ValueError('Snapshot directory absent; restore validated retained links separately, do not guess')
 files={str(ref.relative_to(cache)):sha(ref)};memo={}
 for p in sorted(snapshot.rglob('*')):
  if p.is_dir():
   if p.is_symlink():raise ValueError('Linked snapshot directory rejected')
   continue
  resolved=p.resolve(strict=True)
  if not resolved.is_relative_to((root/'blobs').resolve()) or not resolved.is_file():raise ValueError('Snapshot target must remain inside selected model blobs')
  if resolved not in memo:memo[resolved]=sha(resolved)
  files[str(p.relative_to(cache))]=memo[resolved]
 if len(files)<2:raise ValueError('Empty model snapshot')
 data={}
 for name,wanted in DATA.items():
  p=repo/name
  if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(repo) or sha(p)!=wanted:raise ValueError('Public data hash mismatch: '+name)
  data[name]=wanted
 return {'model_id':'Qwen/Qwen2.5-1.5B-Instruct','revision':revision,'files':files},data
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--repo',required=True);p.add_argument('--out',required=True);a=p.parse_args();manifest,data=build(a.cache,a.repo);out=Path(a.out)
 with out.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
 print(json.dumps({'model_cache_manifest':str(out),'model_cache_manifest_sha256':sha(out),'data_files':data},indent=2))
