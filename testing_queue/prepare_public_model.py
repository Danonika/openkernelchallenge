#!/usr/bin/env python3
"""USER-INVOKED public model bootstrap only; never called by queue/runner or during timing."""
import argparse,os
from pathlib import Path
MODEL='Qwen/Qwen2.5-1.5B-Instruct';REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--confirm-public-download',action='store_true',required=True);a=p.parse_args()
 cache=Path(a.cache).absolute();cache.mkdir(parents=True,exist_ok=True)

 from huggingface_hub import snapshot_download
 path=Path(snapshot_download(repo_id=MODEL,revision=REVISION,cache_dir=str(cache),token=False)).resolve()
 root=cache/'models--Qwen--Qwen2.5-1.5B-Instruct'
 if path!=root.resolve()/'snapshots'/REVISION:raise RuntimeError('Unexpected downloaded snapshot path')
 ref=root/'refs/main';ref.parent.mkdir(exist_ok=True)
 if ref.exists():
  if ref.is_symlink() or ref.read_text().strip()!=REVISION:raise RuntimeError('Existing main ref differs; no overwrite. Select a new dedicated cache.')
 else:
  with ref.open('x') as f:f.write(REVISION)
 print('Public model downloaded at pinned revision. Run prepare_asset_manifest.py and retain its hash before any worker execution.')
if __name__=='__main__':main()
