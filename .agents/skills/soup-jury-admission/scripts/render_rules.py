#!/usr/bin/env python3
"""Generate the table-link checklist from the single canonical rule contract."""
from pathlib import Path
import json
D=Path(__file__).resolve().parents[1]/'references';r=json.loads((D/'rules-v1.json').read_text())
lines=['# Soup jury checklist',f"**Canonical version: {r['rule_version']}** · [machine-readable rules and evidence schema](rules-v1.json)",'',
'This generated view is a checklist, not official admission. Missing evidence is UNKNOWN; an explicit contradiction is FAIL. A PASS requires the referenced proof. Public evaluation cannot prove hidden quality. The five mandatory sections and four submission deliverable categories are all represented below.','',
'## Never confuse these thresholds','',
'| Quantity | Canonical limit | Meaning |','|---|---:|---|',
f"| Useful own Triton share | ≥ {100*r['thresholds']['own_share_min']:g}% in every measured run | Event time / timed-loop wall time from step3; NOT loss |",
f"| Evaluation loss worsening | ≤ {100*r['thresholds']['quality_worsening_max']:g}% | Hidden-row final gate; public only self-check |",
f"| Time / peak growth | ≤ {100*(r['thresholds']['time_growth_max']-1):.0f}% each | Whole command / sampled max memory |",
f"| First training loss gap | ≤ {100*r['thresholds']['first_loss_gap_max']:g}% | Relative to baseline first step |",
f"| Loss trajectory, entropy, accuracy, rewind | ≤ {100*r['thresholds']['curve_gap_max']:g}% | Mean per-element relative gap, not gap of means |",
f"| Speed / memory scoring floor | ratio ≥ {r['thresholds']['scoring_ratio_floor']} | Below this is a scoring tie; separate5% concept |",'',
'## Source discrepancy — visible, not silently resolved','',r['aggregation']['discrepancy'],'',
'Own-share uses common eligible names launched in every solution run and the minimum per-run share. Exclude starter/renamed starter, copied or ready-made kernels, Inductor kernels, unused work and manufactured slowdown. Total useful own share is collective; each individual candidate need not reach the minimum alone.','',
'## Deterministic evidence checklist','', '| ID | Gate | Required proof | Source |','|---|---|---|---|']
for x in r['rules']:lines.append(f"| {x['id']} | {x['group']} | {x['requirement']} | [pinned source]({x['source']}) |")
lines += ['', '## How to use','',
'Create evidence JSON following `rules-v1.json.evidence_schema`; point `result_path` and `checks_path` at immutable local artifacts and provide the exact solution hash. Reviewed `facts` contain a boolean value plus nonempty artifact references. The helper does not independently certify those human/source claims. Missing facts must be omitted or left unknown, never guessed true.','',
'`python ../scripts/checklist.py evidence.json --out checklist.json --markdown checklist.md`','',
'Run the unchanged official harness/checker separately under authorized clean conditions. This helper is read-only and performs no GPU, network, installation, submission or publication. Synthetic examples test the reviewer logic, not challenge compliance.','',
'## Final package, not just kernel code','',
'Platform-created team repository; solution/apply; README with mechanism/source/run/results/limitations and notebook/final commit; completed root SUBMISSION.md; authorized public saved template notebook whose COMMIT matches; demo within the canonical duration limit; presentation. Team size/one-result/deadline and final jury review still apply. Do not publish a private notebook/account identifier in research notes merely because the eventual final notebook must be public.','',
'Official deadline: '+r['deadline']['utc']+' ('+r['deadline']['astana']+'). Current main must be rechecked before finalization. No skill or checklist grants external mutation or disclosure authority.']
(D/'CHECKLIST.md').write_text('\n'.join(lines)+'\n')
