"""Non-executing structural review of candidate notebooks and solution bundles."""
import ast, hashlib, json, pathlib, zipfile

ROOT = pathlib.Path(__file__).resolve().parent
LANES = ('ce', 'head', 'memory', 'nf4', 'lora', 'rotary')

def review(lane):
    root = ROOT / lane
    result = {'lane': lane, 'errors': [], 'notebooks': [], 'bundles': [], 'gpu_validation': 'NOT_RUN'}
    if not root.is_dir():
        result['errors'].append('Lane directory not yet present')
        return result
    for p in root.rglob('*.py'):
        try: ast.parse(p.read_text(), filename=str(p))
        except Exception as exc: result['errors'].append(f'{p.relative_to(root)}: {exc}')
    for p in root.glob('*.ipynb'):
        try:
            doc = json.loads(p.read_text())
            assert doc['nbformat'] == 4
            count = 0
            for cell in doc['cells']:
                if cell['cell_type'] != 'code': continue
                count += 1
                assert not cell.get('outputs'), 'Notebook contains outputs'
                assert cell.get('execution_count') is None, 'Notebook execution count is not clear'
                source = cell['source']
                source = ''.join(source) if isinstance(source, list) else source
                ast.parse(source, filename=f'{p.name}:cell{count}')
            result['notebooks'].append({'path': str(p.relative_to(root)), 'code_cells': count, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        except Exception as exc: result['errors'].append(f'{p.relative_to(root)}: {exc}')
    for p in root.glob('*.zip'):
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                assert len(names) == len(set(names)), 'Duplicate ZIP members'
                for name in names:
                    member = pathlib.PurePosixPath(name)
                    assert not member.is_absolute() and '..' not in member.parts
                    assert name.startswith('solution/'), f'Unexpected member {name}'
                    if name.endswith('/'): continue
                    local = root / name
                    assert local.is_file(), f'Missing matching local source {name}'
                    assert local.read_bytes() == z.read(name), f'ZIP source mismatch {name}'
                result['bundles'].append({'path': p.name, 'files': len(names), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        except Exception as exc: result['errors'].append(f'{p.name}: {exc}')
    if not result['notebooks']: result['errors'].append('No notebook yet')
    if not result['bundles']: result['errors'].append('No solution bundle yet')
    result['structural_status'] = 'PASS' if not result['errors'] else 'INCOMPLETE_OR_FAIL'
    return result

if __name__ == '__main__':
    report = {'scope': 'Syntax, clear notebook outputs, ZIP identity only; not execution or correctness proof', 'lanes': [review(lane) for lane in LANES]}
    print(json.dumps(report, indent=2))
