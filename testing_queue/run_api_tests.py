"""Source-bound local HTTP tests: ephemeral loopback and TEST_ONLY credentials."""
import hashlib,json,platform,unittest
from pathlib import Path
import test_api_unit
P=Path(__file__).resolve().parent
with (P/'api_unit_verified.log').open('w') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_api_unit))
files=['soup_queue.py','api.py','test_api_unit.py','web/dashboard.html','web/dashboard.js','web/dashboard.css','api_unit_verified.log']
out={'scope':'Ephemeral loopback HTTP, dummy credentials only; no public listener/deployment/artifact execution','tests_run':result.testsRun,'failures':[str(t) for t,_ in result.failures],'errors':[str(t) for t,_ in result.errors],'successful':result.wasSuccessful(),'python':platform.python_version(),'sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}}
(P/'api_test_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(not result.wasSuccessful())
