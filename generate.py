import sys

import torch
import torch.nn.functional as F
from cangjie_llm import tokenizer, LLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = tokenizer()
llm = LLM()
checkpoint_path = "./src/cangjie_llm/best_val.pt"
prompt = ""
decode_strategy = "top_k"
temperature = 0.8
top_k = 10
repetition_penalty = 1.15
max_token = 50
allow_byte_tokens = False
stop_on_eos = True


def output_id_for_token(token_name):
    token_id = tok.vocab.get(token_name)
    if token_id is None:
        return None
    output_id = token_id - 26
    return output_id if output_id >= 0 else None


def apply_sampling_filters(logits, generated_ids):
    banned_ids = [
        output_id_for_token("[PAD]"),
        output_id_for_token("[BOS]"),
        output_id_for_token("[UNK]"),
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

    if top_k and 0 < top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k, dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    return logits


def encode_prompt_to_output_ids(prompt_text):
    if not prompt_text:
        return []
    prompt_tokens = tok.tokenize(prompt_text)
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    return llm.head.input_to_output_idx(prompt_tensor).tolist()


ckpt = torch.load(checkpoint_path, map_location=device)
state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
if any(key.startswith("_orig_mod.") for key in state_dict):
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
llm.load_state_dict(state_dict)
llm.to(device)
llm.eval()

tok_id_list = [2]
tok_id_list.extend(encode_prompt_to_output_ids(prompt))
all_vocab = tok.all_vocab().tolist()
eos_id = output_id_for_token("[EOS]")
last_text = ""


with torch.inference_mode():
    for _ in range(max_token):
        tok_list = torch.tensor(
            [all_vocab[tok_id] for tok_id in tok_id_list],
            dtype=torch.long,
            device=device,
        ).view(1, -1, 5)
        next_tok, _ = llm(tok_list)
        next_tok = next_tok[:, -1, :]
        next_tok = apply_sampling_filters(next_tok, tok_id_list)
        if decode_strategy == "greedy":
            next_tok_id = next_tok.argmax(dim=-1).item()
        elif decode_strategy in {"top-k", "top_k"}:
            probs = F.softmax(next_tok / temperature, dim=-1)
            next_tok_id = torch.multinomial(probs, num_samples=1).item()
        else:
            raise ValueError(f"Unsupported decode_strategy: {decode_strategy}")
        if stop_on_eos and next_tok_id == eos_id:
            break
        tok_id_list.append(next_tok_id)
        current_text = tok.detokenize(tok_id_list).removeprefix("[BOS]")
        if current_text.startswith(last_text):
            delta = current_text[len(last_text):]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        else:
            if last_text:
                sys.stdout.write("\n")
            sys.stdout.write(current_text)
            sys.stdout.flush()
        last_text = current_text

if last_text:
    sys.stdout.write("\n")
