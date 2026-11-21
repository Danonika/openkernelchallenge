"""Run the new CUDA-copy probe, then unchanged exploratory composition checks."""
import json
from pathlib import Path
import sys
from .activation_probe import probe
from .exploratory_preflight import main


if __name__ == '__main__':
    # The web supervisor already supplies this canonical --out argument.
    assert len(sys.argv) == 3 and sys.argv[1] == '--out'
    result = probe()
    output = Path(sys.argv[2]).with_name('activation_offload_probe.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('SOUP_ACTIVATION_OFFLOAD_PROBE '+json.dumps(result, sort_keys=True), flush=True)
    main()
