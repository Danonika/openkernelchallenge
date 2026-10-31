"""Execute new infrastructure tests and emit a source-bound JSON summary."""
import hashlib,json,platform,unittest
from pathlib import Path
import test_queue_unit
P=Path(__file__).resolve().parent
with (P/'unit_verified.log').open('w') as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_queue_unit))
out={'scope':'Infrastructure fake-runner tests only; no submitted artifact/scientific/GPU execution','tests_run':result.testsRun,'failures':[str(t) for t,_ in result.failures],'errors':[str(t) for t,_ in result.errors],'successful':result.wasSuccessful(),'python':platform.python_version(),'source_sha256':hashlib.sha256((P/'soup_queue.py').read_bytes()).hexdigest(),'tests_sha256':hashlib.sha256((P/'test_queue_unit.py').read_bytes()).hexdigest(),'log_sha256':hashlib.sha256((P/'unit_verified.log').read_bytes()).hexdigest()}
(P/'test_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(not result.wasSuccessful())
