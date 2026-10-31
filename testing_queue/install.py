#!/usr/bin/env python3
"""Verify release manifest and copy application to a NEW prefix. No root/network/service actions."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path,PurePosixPath
def verify(root):
 root=Path(root).resolve();m=json.loads((root/'PACKAGE_MANIFEST.json').read_text());files=m['files']
 for name,wanted in files.items():
  p=PurePosixPath(name)
  if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('Unsafe manifest path')
  src=root/name
  if any(x.is_symlink() for x in (src,*src.parents)) or not src.is_file():raise ValueError('Missing or linked package file: '+name)
  if hashlib.sha256(src.read_bytes()).hexdigest()!=wanted:raise ValueError('Package hash mismatch: '+name)
 return files
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix');p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parent
 try:
  files=verify(root)
  if not a.verify_only:
   if not a.prefix:raise ValueError('--prefix or --verify-only required')
   dest=Path(a.prefix).absolute()
   if dest.exists():raise ValueError('Prefix exists; choose NEW path. No overwrite/delete is performed')
   dest.mkdir(mode=0o755,parents=True)
   for name in [*files,'PACKAGE_MANIFEST.json']:
    target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target);target.chmod(0o644)
  print(json.dumps({'verified_files':len(files),'installed':not a.verify_only,'prefix':a.prefix if not a.verify_only else None}))
 except (ValueError,OSError,KeyError) as e:print(str(e),file=sys.stderr);raise SystemExit(2)
