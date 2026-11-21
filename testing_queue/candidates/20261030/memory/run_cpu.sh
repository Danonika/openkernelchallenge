#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYTHON="${SOUP_CPU_PYTHON:-/redacted/local/soup_cloud_env/venv/bin/python}"
"$PYTHON" test_cpu_integration.py
"$PYTHON" test_version_sentinel.py
"$PYTHON" test_runner_evidence.py
"$PYTHON" test_process_cleanup.py
"$PYTHON" test_active_quarantine.py
