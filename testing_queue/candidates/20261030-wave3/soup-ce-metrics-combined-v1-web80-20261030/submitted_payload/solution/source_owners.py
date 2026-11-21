"""Pinned constructor-free installation on Soup's cached native Rewind class."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from importlib import metadata
from .chain import Owners

EXPECTED = {
    "tuner_hooks":"2a8f92d6843547416d393d2369289f41d1f259af4bb4b525b61dbe0a3390ae4e",
    "peft_hooks":"13729ed792b82472e019a0ffe6438cfe4a0df955937678b6480b33ff953f510c",
    "alora_offsets":"e97acf9d2591c48c6a7708344caadf5aab5a7a783022f402ddc100cb27f880c8",
    "can_return_tuple":"0c0b7a6197b3a41692415a4db8a5cc85397fc46968697af7090ba34edac89920",
    "rewind_token_rows": "0fb6a78e506af7e76786a4d4d0d64052fa072d4306682f1c86efbd0e779fd81c",
    "rewind_factory": "a87a95be6b050970bf6b2808f073090be1f3da17aa1f97716788bb5294832d08",
    "qwen_init": "baba1570de0e1bc9e6bae18659724b92f46ec3ca4338dbbc45f8f0538161eb01",
    "loss_get": "9fd33976e3cc524193d9d742971d0d5c396e98e3e6fcaa6722da3cff2d537837",
    "loss_set": "4e1501862a200a69fc6ad29fb2a9c9be10d150b0a72d15657d469f2603fc3544",
    "autocast_decorator": '3971b87717ecd0c94581cdd4dfb980367c71a47a900f24145437d116e35633dc',
    'entropy': 'cfeed25ed40872baac62ba7748bf4a43e994377da6236ac84b3b99252702fc93',
    'rewind_rows': '4333291b631bed2709e90a74ecb48a6623f72729e6c3179f3578c385278725f7',
    'converter_call': 'f0a9ee620a17019d1b5362f478c1f157d1ea3c5363fa145cddfeacd60cc96e93',
    'converter_float': '3e4898f124b8573bbd03f09b09cbd2618a1a20dadaa9ef1fafe30c265ad8cdbc',
    'converter_factory': 'ef50cb4a7d8b77d275e9c77ec38fc8b989351d6eed4af0ba788e6d4d341bdd5d',
    'converter_recurse': 'f2c243ce9c8b70287254703a3adb4f9fade26bd9fc9106e505aee2236cf83b20',
    'tuner': '2aedb2eaf5557fa69576193d4bf6d0bf1b56b86a3e7275e3caa9f6d20aa56cc0',
    'get_base_model': 'd4e5baa9c5da5bae4720c8ce828315da0099d41d42688660b0ec684cb6fa630d',
    "qwen": "e7cc0724264fb97ab48ea8e9beeee6edadc87d486085501b2703576b2b2bc183",
    "native_loss": "c7f950f4eb7eba01f9339bc4937c9e1cc2885cf1c1bc116d764990c694a90ff9",
    "rewind": "e05038c89369901e08f5fd46c9fa3dd312e2a0afc8c4eb3d54a3c79eb1fa716a",
    "sft": "949aa5ebf4ec1e6a69c59a958effc389963427174d4fdb5bac6fc02606dc4fb0",
    "hf": "784a390428b9dabb94c4fa2029be628f748a31baaf582d8c437d18e9aa673972",
    "peft": "739d74fe48e688aabd241921d2578bc57add6f6e239dc1b07f66c1e4a1bd864b",
}


def verify_callable(name, fn):
    """Hash native source and reject added wrappers even when wraps hides source."""
    observed=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if observed!=EXPECTED[name]:
        raise RuntimeError("Pinned callable source differs: "+name+" observed_sha256="+observed)
    wrappers=0;cursor=fn
    while hasattr(cursor,"__wrapped__"):
        wrappers+=1;cursor=cursor.__wrapped__
    if wrappers!=(1 if name in ("qwen","rewind_factory","peft_hooks","tuner_hooks") else 0):
        raise RuntimeError("Foreign callable wrapper: "+name)


def load_pinned_owners():
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers import Trainer
    from transformers.modeling_utils import PreTrainedModel
    from trl.trainer.sft_trainer import SFTTrainer, entropy_from_logits
    from peft import PeftModelForCausalLM
    from peft.tuners.lora.model import LoraModel
    from peft.tuners.lora.variants import get_alora_offsets_for_forward
    from transformers.utils.generic import can_return_tuple
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from soup_cli.trainer.rewind_hf import make_rewind_trainer_class, row_losses, row_losses_from_token_losses
    from accelerate.utils.operations import ConvertOutputsToFp32, convert_outputs_to_fp32, convert_to_fp32, recursively_apply
    from torch.amp.autocast_mode import autocast_decorator
    for package,version in {"transformers":"5.16.1","peft":"0.20.0","trl":"0.29.1",
                            "soup-cli":"0.77.0","accelerate":"1.14.0"}.items():
        if metadata.version(package)!=version:
            raise RuntimeError("Pinned dependency differs: "+package)
    if metadata.version("torch").split("+",1)[0]!="2.11.0" or metadata.version("triton")!="3.6.0":
        raise RuntimeError("Pinned Torch/Triton dependency differs")
    cls=make_rewind_trainer_class(SFTTrainer)
    callables={"tuner_hooks":LoraModel._enable_peft_forward_hooks,"peft_hooks":PeftModelForCausalLM._enable_peft_forward_hooks,
               "alora_offsets":get_alora_offsets_for_forward,"can_return_tuple":can_return_tuple,"qwen":Qwen2ForCausalLM.forward,"native_loss":ForCausalLMLoss,
               "rewind":cls.compute_loss,"sft":SFTTrainer.compute_loss,
               "hf":Trainer.compute_loss,"peft":PeftModelForCausalLM.forward,
               "entropy":entropy_from_logits,"rewind_rows":row_losses,
               "tuner":LoraModel.forward,"get_base_model":PeftModelForCausalLM.get_base_model,
               "converter_call":ConvertOutputsToFp32.__call__,"converter_float":convert_to_fp32,
               "converter_factory":convert_outputs_to_fp32,"converter_recurse":recursively_apply,
               "autocast_decorator":autocast_decorator,"rewind_token_rows":row_losses_from_token_losses,
               "rewind_factory":make_rewind_trainer_class,"qwen_init":Qwen2ForCausalLM.__init__,
               "loss_get":PreTrainedModel.loss_function.fget,"loss_set":PreTrainedModel.loss_function.fset}
    for name,fn in callables.items():verify_callable(name,fn)
    linear=torch.nn.Linear.forward
    if (linear.__module__!="torch.nn.modules.linear" or linear.__qualname__!="Linear.forward"
            or hasattr(linear,"__wrapped__") or linear.__code__.co_filename!=inspect.getsourcefile(torch.nn.Linear)):
        raise RuntimeError("Foreign Linear forward owner")
    # Also verify the native single-expression body; a class-method replacement
    # must not be approved merely by comparison with that same class attribute.
    node=ast.parse(textwrap.dedent(inspect.getsource(linear))).body[0]
    body=node.body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]  # Torch2.11 has a non-executable native docstring.
    if (len(body)!=1 or not isinstance(body[0],ast.Return)
            or ast.unparse(body[0].value)!="F.linear(input, self.weight, self.bias)"):
        raise RuntimeError("Foreign Linear forward body")
    owners=Owners(cls,SFTTrainer,Trainer,PeftModelForCausalLM,Qwen2ForCausalLM,LoraModel,
        cls.compute_loss,SFTTrainer.compute_loss,Trainer.compute_loss,
        PeftModelForCausalLM.forward,Qwen2ForCausalLM.forward,LoraModel.forward,
        ForCausalLMLoss,entropy_from_logits,row_losses,PeftModelForCausalLM.get_base_model,linear,ConvertOutputsToFp32,
        ConvertOutputsToFp32.__call__,convert_to_fp32,recursively_apply,
        convert_outputs_to_fp32(lambda:None).__code__,
        autocast_decorator(None,lambda:None).__code__)
    qwen=Qwen2ForCausalLM.forward
    if (qwen.__code__ is not can_return_tuple(lambda:None).__code__
            or inspect.getclosurevars(qwen).nonlocals!={"func":qwen.__wrapped__}
            or qwen.__wrapped__.__globals__.get("CausalLMOutputWithPast") is not CausalLMOutputWithPast):
        raise RuntimeError("Foreign Qwen output wrapper or constructor")
    for fn in [PeftModelForCausalLM._enable_peft_forward_hooks,LoraModel._enable_peft_forward_hooks]:
        if (fn.__code__ is not contextmanager(lambda:None).__code__
                or inspect.getclosurevars(fn).nonlocals!={"func":fn.__wrapped__}):
            raise RuntimeError("Foreign PEFT hook contextmanager wrapper")
    extras={"tuner_hooks":LoraModel._enable_peft_forward_hooks,"peft_hooks":PeftModelForCausalLM._enable_peft_forward_hooks,
            "alora_offsets":get_alora_offsets_for_forward,"output_class":CausalLMOutputWithPast}
    return owners, row_losses_from_token_losses, extras
