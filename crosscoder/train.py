from utils import *
from trainer import Trainer

tokens_no_bos = compile_all_tokens(save_dir="/data2/hluo/checkpoints/crosscoders", model_name="Qwen/Qwen2-1.5B")
tokens_with_bos = torch.cat([torch.zeros(tokens_no_bos.shape[0], 1, device=tokens_no_bos.device, dtype=torch.int32), tokens_no_bos[:, :-1]], dim=1)

SAVE_DIR="/data2/hluo/checkpoints/crosscoders/qwen_layer_24_27"

device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2-1.5B", 
    device=device, 
)
# base_model = HookedTransformer.from_pretrained(
#     "Qwen/Qwen2-1.5B-Instruct", 
#     device=device, 
# )
# chat_model = HookedTransformer.from_pretrained(
#     "Qwen/Qwen2-1.5B", 
#     device=device, 
# )

default_cfg = {
    "seed": 49,
    "batch_size": 2048,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 25_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 256,
    "enc_dtype": "fp32",
    "model_name": "qwen",
    "site": "resid_pre",
    "device": device,
    "model_batch_size": 16,
    "log_every": 20,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "hook_resid_pre",
    "layer_length": (24, 27),
    "wandb_project": "crosscoder",
    "wandb_run_name": "qwen_multi_layer_24_27",
    "save_dir": SAVE_DIR,
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(
    cfg, 
    base_model,
    tokens_with_bos
)
# trainer = Trainer(
#     cfg, 
#     base_model, 
#     tokens_with_bos
#     chat_model, 
# )
trainer.train()