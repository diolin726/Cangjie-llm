import torch
import torch.nn.functional as F
from contextlib import nullcontext
from cangjie_converter import get_char_cangjie_tokens
from cangjie_tokenizer import CangjieLlamaTokenizer, CJ_TOKEN_TO_ID, is_chinese_char
from cangjie_detokenizer import CangjieCosineDetokenizer
from cangjie_tokenizer import CangjieLlamaTokenizer as _TokClass


from modeling_cangjie_llama import CangjieTextGenerationModel, TOTAL_VOCAB_SIZE, create_causal_mask

def generate_autoregressive(model, tokenizer, detokenizer, prompt, max_new_tokens=40, temperature=0.7, device='cuda'):
    """
    自迴歸生成 (Autoregressive Text Generation):
    輸入序列 "abc" -> 模型預測下一個字 "d" -> 序列變為 "abcd" -> 預測下一個字 "e"...
    
    中文字解碼：採用構想 C (餘弦相似度矩陣匹配 Cosine Similarity Retrieval)
    """
    # Use TokenizerCache to avoid re-encoding entire prompt every step
    cache = tokenizer.TokenizerCache.from_text(tokenizer, prompt)
    std_ids_t, mask_t, cj_ids_t = cache.to_tensors()
    std_ids_t = std_ids_t.unsqueeze(0).to(device)
    mask_t = mask_t.unsqueeze(0).to(device)
    cj_ids_t = cj_ids_t.unsqueeze(0).to(device)

    print(f"\n[初始 Prompt]: '{prompt}' (len={std_ids_t.shape[1]})")

    # First full pass to obtain initial hidden states and logits
    with torch.no_grad():
        amp = torch.cuda.amp.autocast if hasattr(torch.cuda.amp, 'autocast') else nullcontext
        with amp(enabled=(device.startswith('cuda'))):
            x = model.embedding(std_ids_t, mask_t, cj_ids_t)
            seq_len = std_ids_t.shape[1]
            causal_mask = create_causal_mask(seq_len, std_ids_t.device).unsqueeze(0)
            for decoder in model.decoders:
                x = decoder(x, causal_mask, model.rope)
            x = model.norm(x)
            logits = model.out(x)

    generated = ''
    # keep past_hidden for incremental updates
    past_hidden = x  # (B, L, H)

    for step in range(max_new_tokens):
        # compute logits for last token from past_hidden
        logits_last = model.out(past_hidden)[0, -1, :]

        bpe_logits = logits_last[:10000]
        probs = F.softmax(bpe_logits / temperature, dim=-1)
        max_bpe_prob = probs.max().item()

        if max_bpe_prob > 0.6:
            next_bpe_id = torch.multinomial(probs, num_samples=1).item()
            if next_bpe_id == 1:
                break
            next_char = tokenizer.base_tokenizer.decode([next_bpe_id])
            cache.append_bpe_id(next_bpe_id)
        else:
            # detokenize using last hidden vector
            h_last = past_hidden[0, -1, :]
            next_char = detokenizer.decode_vector(h_last, temperature=temperature)
            cache.append_chinese_char(next_char)

        # incremental: compute new embedding for just the appended token
        std_ids_t2, mask_t2, cj_ids_t2 = cache.to_tensors()
        # get the last token tensors
        last_std = std_ids_t2[-1:].unsqueeze(0).to(device)  # (1,1)
        last_mask = mask_t2[-1:].unsqueeze(0).to(device)
        last_cj = cj_ids_t2[-1:].unsqueeze(0).to(device)  # (1,1,5)

        with torch.no_grad():
            with amp(enabled=(device.startswith('cuda'))):
                new_emb = model.embedding(last_std, last_mask, last_cj)  # (1,1,H)
                # run each decoder in cache mode on the new token
                new_h = new_emb
                for decoder in model.decoders:
                    # decoder accepts use_cache flag in fallback implementation
                    try:
                        new_h = decoder(new_h, None, model.rope, use_cache=True)
                    except TypeError:
                        new_h = decoder(new_h, None, model.rope)
                new_h = model.norm(new_h)

        # append new_h to past_hidden
        past_hidden = torch.cat([past_hidden, new_h], dim=1)
        generated += next_char

    return prompt + generated

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用執行裝置: {device}")
    
    # 1. 載入 Tokenizer 與預訓練模型
    tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
    model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
    
    # 2. 載入訓練好的 Stage 2 Checkpoint 權重
    checkpoint_path = 'cangjie_llama_fineweb_stage2.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. 初始化構想 C 餘弦相似度 Detokenizer (建構 65,176 漢字 Codebook)
    detokenizer = CangjieCosineDetokenizer(model)

    # 4. 測試自迴歸生成 (abc -> d -> abcd -> e...)
    prompts = [
        "Hello World,",
        "Once upon a time,",
        "明月幾時有",
        "高校人才培養"
    ]
    
    print("\n==========================================")
    print(" 測試自迴歸生成 (Autoregressive Generation)")
    print("==========================================")
    for p in prompts:
        output_text = generate_autoregressive(model, tokenizer, detokenizer, p, max_new_tokens=25, temperature=0.7, device=device)
        print(f"[最終生成結果]: {output_text}\n" + "-"*50)
