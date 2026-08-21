import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from cangjie_llm.config import (
        accumulation_steps,
        batch_size,
        best_checkpoint_path,
        block_size,
        checkpoint_interval,
        dataset_cache_path,
        dataset_mix,
        dataset_name,
        dataset_split,
        dataset_streaming,
        device,
        enable_torch_compile,
        epochs,
        get_lr,
        gpu_count,
        grad_clip,
        initial_checkpoint_path,
        latest_checkpoint_path_template,
        log_interval,
        lr,
        num_workers,
        resume_checkpoint_path,
        resume_training,
        sample_max_tokens,
        sample_prompts,
        streaming_shuffle_buffer,
        streaming_steps_per_epoch,
        streaming_text_batch_size,
        torch_compile_mode,
        use_bf16_autocast,
        validation_max_batches,
        validation_ratio,
    )
    from cangjie_llm.dataset import (
        CangjieDataset,
        StreamingCangjieDataset,
        collate_cangjie_batch,
    )
    from cangjie_llm.model import LLM, cj_head, embedding
    from cangjie_llm.tokenization import tokenizer
else:
    from .config import (
        accumulation_steps,
        batch_size,
        best_checkpoint_path,
        block_size,
        checkpoint_interval,
        dataset_cache_path,
        dataset_mix,
        dataset_name,
        dataset_split,
        dataset_streaming,
        device,
        enable_torch_compile,
        epochs,
        get_lr,
        gpu_count,
        grad_clip,
        initial_checkpoint_path,
        latest_checkpoint_path_template,
        log_interval,
        lr,
        num_workers,
        resume_checkpoint_path,
        resume_training,
        sample_max_tokens,
        sample_prompts,
        streaming_shuffle_buffer,
        streaming_steps_per_epoch,
        streaming_text_batch_size,
        torch_compile_mode,
        use_bf16_autocast,
        validation_max_batches,
        validation_ratio,
    )
    from .dataset import CangjieDataset, StreamingCangjieDataset, collate_cangjie_batch
    from .model import LLM, cj_head, embedding
    from .tokenization import tokenizer


def _atomic_torch_save(obj, save_path):
    tmp_path = f"{save_path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, save_path)


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap_model(model).state_dict().items()
    }


@torch.no_grad()
def save_checkpoint(state_dict, save_path):
    def _save():
        _atomic_torch_save(state_dict, save_path)
        print(f"saved to {save_path}")

    thread = threading.Thread(target=_save, daemon=True)
    thread.start()


def load_checkpoint(load_path, map_location):
    state_dict = torch.load(load_path, map_location=map_location)
    normalized_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith("_orig_mod.") or key.startswith("module."):
            if key.startswith("_orig_mod."):
                key = key.removeprefix("_orig_mod.")
            if key.startswith("module."):
                key = key.removeprefix("module.")
        normalized_state_dict[key] = value
    return normalized_state_dict


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def save_resume_checkpoint(model, optimizer, epoch, step, best_val_loss, save_path):
    payload = {
        "model": _cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }
    _atomic_torch_save(payload, save_path)
    print(f"saved resume checkpoint to {save_path}")


def load_resume_checkpoint(load_path, map_location):
    payload = torch.load(load_path, map_location=map_location)
    model_state = payload.get("model", {})
    normalized_state = {}
    for key, value in model_state.items():
        while key.startswith("_orig_mod.") or key.startswith("module."):
            if key.startswith("_orig_mod."):
                key = key.removeprefix("_orig_mod.")
            if key.startswith("module."):
                key = key.removeprefix("module.")
        normalized_state[key] = value
    payload["model"] = normalized_state
    return payload


def output_id_for_token(tok, token_name):
    token_id = tok.vocab.get(token_name)
    if token_id is None:
        return None
    output_id = token_id - 26
    return output_id if output_id >= 0 else None


def apply_generation_filters(logits, tok, generated_ids, repetition_penalty=1.2, allow_byte_tokens=False):
    banned_ids = [
        output_id_for_token(tok, "[PAD]"),
        output_id_for_token(tok, "[BOS]"),
        output_id_for_token(tok, "[UNK]"),
    ]
    for token_id in banned_ids:
        if token_id is not None and token_id < logits.size(-1):
            logits[:, token_id] = float("-inf")

    if not allow_byte_tokens:
        for token_name, token_id in tok.vocab.items():
            if token_name.startswith("<BYTE_"):
                output_id = token_id - 26
                if 0 <= output_id < logits.size(-1):
                    logits[:, output_id] = float("-inf")

    if repetition_penalty != 1.0:
        for token_id in set(generated_ids):
            if 0 <= token_id < logits.size(-1):
                token_logits = logits[:, token_id]
                logits[:, token_id] = torch.where(
                    token_logits > 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty,
                )

    return logits


def encode_prompt_to_output_ids(prompt_text, tok, head, target_device):
    if not prompt_text:
        return []
    prompt_tokens = tok.tokenize(prompt_text)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=target_device)
    return head.input_to_output_idx(prompt_tensor).tolist()


@torch.no_grad()
def evaluate_loss(model, data_loader, max_batches):
    if data_loader is None:
        return None
    was_training = model.training
    model.eval()
    losses = []
    try:
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            y = y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                _, loss = model(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            losses.append(loss.detach().float().item())
    finally:
        if was_training:
            model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample_text(model, tok, prompt_text, max_tokens):
    was_training = model.training
    base_model = unwrap_model(model)
    model.eval()
    try:
        tok_id_list = [2]
        tok_id_list.extend(encode_prompt_to_output_ids(prompt_text, tok, base_model.head, device))
        all_vocab = tok.all_vocab().tolist()
        eos_id = output_id_for_token(tok, "[EOS]")
        for _ in range(max_tokens):
            tok_list = torch.tensor(
                [all_vocab[tok_id] for tok_id in tok_id_list],
                dtype=torch.long,
                device=device,
            ).view(1, -1, 5)
            next_tok, _ = model(tok_list)
            next_tok = apply_generation_filters(next_tok[:, -1, :], tok, tok_id_list)
            next_tok_id = next_tok.argmax(dim=-1).item()
            if eos_id is not None and next_tok_id == eos_id:
                break
            tok_id_list.append(next_tok_id)
        return tok.detokenize(tok_id_list).removeprefix("[BOS]")
    finally:
        if was_training:
            model.train()


def create_optimizer(model):
    if device == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), lr, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr)


def maybe_enable_multi_gpu(model):
    if gpu_count <= 1:
        return model
    device_ids = list(range(gpu_count))
    print(f"啟用 DataParallel: GPUs={device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def maybe_compile_model(model):
    if not enable_torch_compile:
        if device == "cuda" and gpu_count > 1:
            print("多 GPU 模式下停用 torch.compile，避免與 DataParallel 衝突")
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile 不可用，使用 eager mode")
        return model
    try:
        compiled_model = torch.compile(model, mode=torch_compile_mode)
        print(f"torch.compile 已啟用: mode={torch_compile_mode}")
        return compiled_model
    except Exception as exc:
        print(f"torch.compile 啟用失敗，退回 eager mode: {exc}")
        return model


def main():
    model = LLM()
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"總參數量: {total_params:>12,} ({total_params / 1e6:.2f} M)")
    print(f"可訓練參數量: {trainable_params:>12,} ({trainable_params / 1e6:.2f} M)")
    print(f"凍結參數量:{total_params - trainable_params:>12,}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<40} | {param.numel():,}")
    head = cj_head(embedding(1, 1))
    print(head.input_to_output_idx(torch.tensor([
        [30, 26, 26, 26, 26],
        [7, 14, 20, 20, 10],
        [89, 26, 26, 26, 26],
        [99, 26, 26, 26, 26],
        [1, 7, 13, 5, 26],
        [7, 0, 15, 8, 26],
        [44, 26, 26, 26, 26],
        [96, 26, 26, 26, 26],
        [18, 12, 7, 0, 26],
        [64, 26, 26, 26, 26],
        [255, 26, 26, 26, 26],
        [12, 6, 1, 26, 26],
        [91, 26, 26, 26, 26],
        [56, 26, 26, 26, 26],
        [21, 9, 7, 22, 26],
        [13, 1, 18, 7, 16],
        [45, 26, 26, 26, 26],
        [24, 17, 18, 20, 26],
        [30, 26, 26, 26, 26],
        [24, 2, 10, 26, 26],
        [7, 14, 12, 12, 13],
        [24, 17, 16, 12, 1],
        [3, 0, 7, 20, 26],
        [14, 7, 16, 26, 26],
        [42, 26, 26, 26, 26],
        [7, 16, 15, 7, 7],
        [19, 22, 3, 26, 26],
        [50, 26, 26, 26, 26],
        [24, 19, 0, 9, 26],
        [113, 26, 26, 26, 26],
        [0, 26, 26, 26, 26],
        [13, 23, 20, 26, 26],
    ], dtype=torch.int16)))

    if dataset_streaming:
        train_source = StreamingCangjieDataset(
            dataset_name=dataset_name,
            dataset_mix=dataset_mix,
            split=dataset_split,
            block_size=block_size,
            shuffle_buffer=streaming_shuffle_buffer,
            text_batch_size=streaming_text_batch_size,
        )
        val_source = None
        effective_num_workers = 0
        print(
            "啟用 streaming dataset："
            f" steps_per_epoch={streaming_steps_per_epoch:,},"
            f" shuffle_buffer={streaming_shuffle_buffer:,},"
            f" text_batch_size={streaming_text_batch_size}"
        )
        print("Streaming 模式下停用 validation split；將只記錄 training loss")
    else:
        train_ds = CangjieDataset(
            dataset_name=dataset_name,
            dataset_mix=dataset_mix,
            split=dataset_split,
            block_size=block_size,
            cache_path=dataset_cache_path,
        )

        dataset_len = len(train_ds)
        tentative_val_size = min(
            max(batch_size, int(dataset_len * validation_ratio)),
            batch_size * validation_max_batches,
        )
        val_size = tentative_val_size if dataset_len > tentative_val_size else 0
        train_size = dataset_len - val_size
        if val_size > 0:
            train_source = Subset(train_ds, range(train_size))
            val_source = Subset(train_ds, range(train_size, dataset_len))
            print(f"資料切分: train={len(train_source):,} | val={len(val_source):,}")
        else:
            train_source = train_ds
            val_source = None
            print("資料量不足以建立 validation split，將只記錄 training loss")
        effective_num_workers = num_workers

    train_loader = DataLoader(
        train_source,
        batch_size,
        shuffle=not dataset_streaming,
        collate_fn=collate_cangjie_batch,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=4 if effective_num_workers > 0 else None,
    )
    val_loader = None
    if val_source is not None:
        val_loader = DataLoader(
            val_source,
            batch_size,
            shuffle=False,
            collate_fn=collate_cangjie_batch,
            num_workers=0,
            pin_memory=True,
        )
    for batch, target in train_loader:
        print("Batch shape:", batch.shape)
        print("Target shape", target.shape)
        break

    if resume_training and os.path.exists(resume_checkpoint_path):
        print(f"resuming training from {resume_checkpoint_path}")
        resume_state = load_resume_checkpoint(resume_checkpoint_path, map_location=device)
        model.load_state_dict(resume_state["model"])
    elif initial_checkpoint_path and os.path.exists(initial_checkpoint_path):
        state_dict = load_checkpoint(initial_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        resume_state = None
    elif initial_checkpoint_path:
        print(
            f"initial checkpoint not found: {initial_checkpoint_path} | "
            "start from scratch"
        )
        resume_state = None
    else:
        resume_state = None
    model.to(device)
    model = maybe_enable_multi_gpu(model)
    model = maybe_compile_model(model)
    optimizer = create_optimizer(model)
    start_epoch = 0
    start_step = 0
    best_val_loss = float("inf")
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = resume_state.get("epoch", 0)
        start_step = resume_state.get("step", 0)
        best_val_loss = resume_state.get("best_val_loss", best_val_loss)
    model.train()
    steps_per_epoch = streaming_steps_per_epoch if dataset_streaming else len(train_loader)
    total_steps = epochs * steps_per_epoch
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        num_batches = steps_per_epoch
        running_loss = None
        interval_steps = 0
        interval_start_time = time.perf_counter()
        print(f"epoch{epoch} starts")
        epoch_start_step = start_step if epoch == start_epoch else 0
        if dataset_streaming and hasattr(train_source, "set_epoch"):
            train_source.set_epoch(epoch)
        if dataset_streaming:
            train_iter = iter(train_loader)
            skipped_steps = 0
            while skipped_steps < epoch_start_step:
                try:
                    next(train_iter)
                except StopIteration:
                    break
                skipped_steps += 1
            step_iterator = range(epoch_start_step, num_batches)
        else:
            step_iterator = enumerate(train_loader)

        for step_item in step_iterator:
            if dataset_streaming:
                step = step_item
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    print("Streaming 資料提早耗盡，提前結束本 epoch")
                    break
            else:
                step, (x, y) = step_item

            global_step = epoch * num_batches + step + 1
            current_lr = get_lr(global_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            x = x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
            y = y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                _, loss = model(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            (loss / accumulation_steps).backward()

            should_step = ((step - epoch_start_step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_for_log = loss.detach()
            running_loss = loss_for_log if running_loss is None else running_loss + loss_for_log
            interval_steps += 1
            is_log_step = (step + 1) % log_interval == 0 or (step + 1) == num_batches
            is_checkpoint_step = (step + 1) % checkpoint_interval == 0 or (step + 1) == num_batches
            if is_log_step:
                elapsed = time.perf_counter() - interval_start_time
                steps_per_sec = interval_steps / elapsed if elapsed > 0 else 0.0
                progress = (step + 1) / num_batches * 100
                avg_loss = (running_loss / interval_steps).item()
                print(
                    f"epoch [{epoch + 1}/{epochs}] | "
                    f"step [{step + 1}/{num_batches}] ({progress:.1f}%) | "
                    f"avg Loss: {avg_loss:.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"steps/s: {steps_per_sec:.2f}"
                )
                val_loss = evaluate_loss(model, val_loader, validation_max_batches)
                if val_loss is not None:
                    print(f"validation Loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(_cpu_state_dict(model), best_checkpoint_path)
                        print(
                            f"saved best validation checkpoint: "
                            f"{best_checkpoint_path} ({best_val_loss:.4f})"
                        )
                tok = tokenizer()
                for prompt_text in sample_prompts:
                    sample = generate_sample_text(model, tok, prompt_text, sample_max_tokens)
                    print(f"sample[{prompt_text or '<empty>'}]: {sample}")
                running_loss = None
                interval_steps = 0
                interval_start_time = time.perf_counter()
            if is_checkpoint_step:
                latest_checkpoint_path = latest_checkpoint_path_template.format(epoch=epoch + 1)
                save_checkpoint(_cpu_state_dict(model), latest_checkpoint_path)
                save_resume_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    step + 1,
                    best_val_loss,
                    resume_checkpoint_path,
                )
        start_step = 0


__all__ = [
    "apply_generation_filters",
    "create_optimizer",
    "encode_prompt_to_output_ids",
    "evaluate_loss",
    "generate_sample_text",
    "load_checkpoint",
    "main",
    "maybe_compile_model",
    "maybe_enable_multi_gpu",
    "output_id_for_token",
    "save_checkpoint",
    "unwrap_model",
]


if __name__ == "__main__":
    main()
