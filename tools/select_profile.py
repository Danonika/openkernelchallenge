"""Select a fixed source profile BEFORE testing or measuring a fresh process.

Profiles change implementation choices only, never task/config/harness/data.
There is no profile selection or tuning inside solution.apply() or training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = {
    "HEAD_MODE": "cached",
    "CACHED_CE_CHUNK_TOKENS": 128,
    "CHUNK_TOKENS": 256,
    "SPARSE_BACKWARD": True,
    "HEAD_PROJECTION": "torch",
    "FUSED_NF4": True,
    "FROZEN_STORAGE": False,
    "COMPACT_MLP_ACTIVATIONS": True,
    "FUSED_ROTARY": False,
    "FUSED_LORA_EPILOGUE": False,
    "FUSED_LORA_DROPOUT": True,
    "SHARED_ATTENTION_MASK": False,
    "FUSED_RMSNORM": True,
}

PROFILES = {
    "fused": {},
    "cached256": {"CACHED_CE_CHUNK_TOKENS": 256},
    "epilogues": {"CACHED_CE_CHUNK_TOKENS": 256,
                  "FUSED_LORA_EPILOGUE": True, "FUSED_ROTARY": True},
    "streamed": {"HEAD_MODE": "streamed"},
    "shared_mask": {"CACHED_CE_CHUNK_TOKENS": 256,
                    "SHARED_ATTENTION_MASK": True},
    "frozen_storage": {"FROZEN_STORAGE": True},
}


def profile_values(name):
    return dict(BASE, **PROFILES[name])


def select_profile(name, repo):
    target = Path(repo) / "solution" / "settings.py"
    if not target.is_file():
        raise FileNotFoundError(f"missing source settings: {target}")
    values = profile_values(name)
    source = ('"""Fixed implementation profile: ' + name + '.\n'
              'Selected before a fresh run. Task settings are unchanged.\n'
              'No on-device or benchmark-dependent profile selection.\n"""\n\n')
    source += "\n".join(f"{key} = {value!r}" for key, value in values.items()) + "\n"
    temporary = target.with_suffix(".py.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(target)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(PROFILES))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--show", action="store_true", help="show values without writing")
    args = parser.parse_args()
    values = profile_values(args.profile) if args.show else select_profile(args.profile, args.repo)
    print(json.dumps({"profile": args.profile, "settings": values,
                      "written": not args.show}, indent=2))


if __name__ == "__main__":
    main()
