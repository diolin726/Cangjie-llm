import math
import os
import re
from pathlib import Path

import torch

torch.manual_seed(67)  # 676767

PACKAGE_DIR = Path(__file__).resolve().parent

# ===== 可調參數 =====
dropout = 0.05
embed_size = 512
ffn_hidden_size = 1344

batch_size = 128
accumulation_steps = 8
grad_clip = 1.0
num_workers = 8
block_size = 256
window_stride = 128
n_head = 8
n_kv_head = 4
n_layer = 10
rope_theta = 1e6
lr = 3e-5
min_lr = 3e-6
warmup_steps = 200
plateau_patience = 3
plateau_factor = 0.5
plateau_min_delta = 0.003
plateau_min_lr = 3e-6
epochs = 1
log_interval = 1000
checkpoint_interval = 1000
torch_compile_mode = "default"
return_training_logits = False
sampled_softmax_negatives = 0
cangjie_auxiliary_loss_weight = 0.15
dataset_name = "yuhuanstudio/wikipedia-zh-tw"
dataset_split = "train"
dataset_cache_path = "./cangjie_cached.pt"
dataset_streaming = False
use_token_shards = False
token_shard_dir = "./cangjie_token_shards"
token_shard_size_mb = 256
token_shard_max_cache_gb = 40
token_shard_validation_shards = 4
dataset_mix = [
    {
        "name": "opencsg/Fineweb-Edu-Chinese-V2.1",
        "weight": 0.2,
        "dataset_dir": "4_5",
        "data_files":["000*"]
    },
    {
        "name": "yuhuanstudio/wikipedia-zh-tw",
        "weight": 0.2
    },
    {
        "name": "yuhuanstudio/OpenNewsArchive_pretrain_zhtw",
        "weight": 0.3
    },
    {
        "name": "agentlans/traditional-chinese",
        "weight": 0.2
    },
    {
        "name": "yuhuanstudio/PTT-pretrain-zhtw",
        "weight": 0.1
    }
]
streaming_shuffle_buffer = 1_024
streaming_steps_per_epoch = 61_507
streaming_text_batch_size = 256
streaming_num_workers = 4
validation_ratio = 0.002
validation_max_batches = 8
sample_prompts = ["" , "台灣最高的山是","路口那家小吃店","1~100裡面我選","以下是中國的省份:"]
sample_max_tokens = 50
resume_training = False
resume_checkpoint_path = str(PACKAGE_DIR / "training_resume.pt")
initial_checkpoint_path = str(PACKAGE_DIR / "best_val.pt")
best_checkpoint_path = str(PACKAGE_DIR / "best_val.pt")
latest_checkpoint_path_template = str(PACKAGE_DIR / "cangjie_epoch_{epoch}_latest.pt")
preprocess_batch_size = 1024
preprocess_chunk_rows = 10_000_000
preprocess_workers = max(1, min(4, os.cpu_count() or 1))
preprocess_queue_depth = max(2, preprocess_workers * 2)
cache_format_version = 3
common_char_table_size = 3500
common_word_table_size = 20000
common_word_max_length = 4
detokenize_beam_size = 4
detokenize_context_window = 12

#auto check vocab_size
from cangjie_convertor import cj_encoder
vocab_size_use = cj_encoder()
vocab_size = vocab_size_use.vocab_size #383

# Remove regex backreference remnants such as ``\1`` or ``\123`` before tokenization.
training_escape_pattern = re.compile(r"\\[1-9][0-9]*")

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
device = "cuda" if gpu_count > 0 else "cpu"
use_bf16_autocast = device == "cuda" and torch.cuda.is_bf16_supported()
enable_torch_compile = device == "cuda" and gpu_count <= 1

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

print(f"using {device} ({gpu_count} GPU{'s' if gpu_count != 1 else ''} visible)")
if device == "cuda":
    print(f"bf16 autocast: {'enabled' if use_bf16_autocast else 'disabled'}")
    print(f"flash sdp: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"mem efficient sdp: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"math sdp fallback: {torch.backends.cuda.math_sdp_enabled()}")


def get_lr(step, total_steps):
    if total_steps <= 0:
        return lr
    cosine = math.cos(math.pi * step / total_steps)
    return lr * (0.1 + 0.45 * (1.0 + cosine))


__all__ = [
    "batch_size",
    "accumulation_steps",
    "best_checkpoint_path",
    "block_size",
    "cache_format_version",
    "cangjie_auxiliary_loss_weight",
    "checkpoint_interval",
    "common_char_table_size",
    "common_word_max_length",
    "common_word_table_size",
    "detokenize_beam_size",
    "detokenize_context_window",
    "dataset_cache_path",
    "dataset_mix",
    "dataset_name",
    "dataset_split",
    "dataset_streaming",
    "device",
    "dropout",
    "embed_size",
    "enable_torch_compile",
    "epochs",
    "ffn_hidden_size",
    "get_lr",
    "grad_clip",
    "gpu_count",
    "initial_checkpoint_path",
    "latest_checkpoint_path_template",
    "log_interval",
    "lr",
    "min_lr",
    "n_head",
    "n_kv_head",
    "n_layer",
    "num_workers",
    "plateau_factor",
    "plateau_min_delta",
    "plateau_min_lr",
    "plateau_patience",
    "preprocess_batch_size",
    "preprocess_chunk_rows",
    "preprocess_queue_depth",
    "preprocess_workers",
    "return_training_logits",
    "resume_checkpoint_path",
    "resume_training",
    "rope_theta",
    "sample_max_tokens",
    "sample_prompts",
    "sampled_softmax_negatives",
    "streaming_shuffle_buffer",
    "streaming_steps_per_epoch",
    "streaming_text_batch_size",
    "streaming_num_workers",
    "torch_compile_mode",
    "token_shard_dir",
    "token_shard_max_cache_gb",
    "token_shard_size_mb",
    "token_shard_validation_shards",
    "training_escape_pattern",
    "use_bf16_autocast",
    "use_token_shards",
    "validation_max_batches",
    "validation_ratio",
    "vocab_size",
    "warmup_steps",
    "window_stride",
]
