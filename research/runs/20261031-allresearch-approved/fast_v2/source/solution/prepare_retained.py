"""Clone exact PEFT preparation with one storage-only allocation exception.

This is original orchestration code, not a Triton kernel. The exact native
function still freezes every base parameter, widens every other eligible
parameter, and preserves checkpoint/cache behavior. Only the attested tied
FP16 table skips a widening allocation; its values never change.
"""
import ast
import hashlib
import inspect

PREPARE_SHA256 = '8a080fa0f22a185031e6ad97f15ed5bb19e79371209066302a34b21696df6df4'
PRIVATE_ARGUMENT = '_soup_retained_tied_parameter'


def make_retained_prepare(native_prepare):
    source = inspect.getsource(native_prepare)
    if hashlib.sha256(source.encode()).hexdigest() != PREPARE_SHA256:
        raise RuntimeError('Exact native PEFT preparation source changed')
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('Unexpected PEFT preparation source structure')
    function = tree.body[0]
    if function.name != 'prepare_model_for_kbit_training' or function.decorator_list:
        raise RuntimeError('Unexpected PEFT preparation function owner')
    if any(node.id == PRIVATE_ARGUMENT for node in ast.walk(function) if isinstance(node, ast.Name)):
        raise RuntimeError('Private storage argument collides with native source')
    expected = ast.dump(ast.parse('param.data = param.data.to(torch.float32)').body[0])

    class KeepOneStorage(ast.NodeTransformer):
        def __init__(self):
            self.replacements = 0

        def visit_Assign(self, node):
            if ast.dump(node) != expected:
                return self.generic_visit(node)
            self.replacements += 1
            condition = ast.Compare(left=ast.Name(id='param', ctx=ast.Load()),
                ops=[ast.IsNot()], comparators=[ast.Name(id=PRIVATE_ARGUMENT, ctx=ast.Load())])
            return ast.copy_location(ast.If(test=condition, body=[node], orelse=[]), node)

    transform = KeepOneStorage()
    tree = transform.visit(tree)
    if transform.replacements != 1:
        raise RuntimeError('Native PEFT widening boundary was not unique')
    function.args.kwonlyargs.append(ast.arg(arg=PRIVATE_ARGUMENT))
    function.args.kw_defaults.append(ast.Constant(value=None))
    ast.fix_missing_locations(tree)
    namespace = dict(native_prepare.__globals__)
    exec(compile(tree, '<soup-retained-native-prepare>', 'exec'), namespace)
    clone = namespace['prepare_model_for_kbit_training']
    if clone.__globals__.get('torch') is not native_prepare.__globals__.get('torch'):
        raise RuntimeError('Native PEFT preparation globals changed')
    return clone
