import torch
import torch.nn.functional as F
from cangjie_converter import get_char_cangjie_tokens
from cangjie_tokenizer import CangjieLlamaTokenizer, CJ_TOKEN_TO_ID, is_chinese_char
from cangjie_detokenizer import CangjieCosineDetokenizer


from modeling_cangjie_llama import CangjieTextGenerationModel, TOTAL_VOCAB_SIZE

def generate_autoregressive(model, tokenizer, detokenizer, prompt, max_new_tokens=40, temperature=0.7, device='cuda'):
    """
    自迴歸生成 (Autoregressive Text Generation):
    輸入序列 "abc" -> 模型預測下一個字 "d" -> 序列變為 "abcd" -> 預測下一個字 "e"...
    
    中文字解碼：採用構想 C (餘弦相似度矩陣匹配 Cosine Similarity Retrieval)
    """
    current_text = prompt
    print(f"\n[初始 Prompt]: '{prompt}'")
    
    for step in range(max_new_tokens):
        # 1. 將當前累積的文字進行雙軌 Encoding
        std_ids, mask, cj_ids = tokenizer.encode(current_text)
        
        std_ids_t = std_ids.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        cj_ids_t = cj_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 2. Forward Pass: 獲取最後一個 Step 的 768 維 H_last 隱層向量與 Logits
            x = model.embedding(std_ids_t, mask_t, cj_ids_t)
            seq_len = std_ids_t.shape[1]
            causal_mask = model.rope.cos.new_ones(seq_len, seq_len, dtype=torch.bool).tril().unsqueeze(0)
            
            for decoder in model.decoders:
                x = decoder(x, causal_mask, model.rope)
            x = model.norm(x)
            
            # 取最後一個位置的 768 維隱層向量 H_last
            h_last = x[0, -1, :]  # (768,)
            
            # 同時獲取原標準 10,027 Logits (前 10,000 個為標準 BPE Token)
            logits_last = model.out(x)[0, -1, :]  # (10027,)

        # 3. 判斷下一個預測 Token 種類：
        # 如果標準 Logits 在英文/標點 (前 10,000) 的最大概率明顯高於倉頡範圍，選擇 BPE Token
        bpe_logits = logits_last[:10000]
        max_bpe_prob = F.softmax(bpe_logits / temperature, dim=-1).max().item()

        if max_bpe_prob > 0.6:  # 生成英文 / 數字 / 標點
            probs = F.softmax(bpe_logits / temperature, dim=-1)
            next_bpe_id = torch.multinomial(probs, num_samples=1).item()
            if next_bpe_id == 1:  # [eos]
                print("遇到 [eos] 終止符，生成結束。")
                break
            next_char = tokenizer.base_tokenizer.decode([next_bpe_id])
        else:
            # 使用構想 C：透過 768 維向量 H_last 與 65,176 Codebook 進行餘弦相似度匹配，獲得下一個中文字！
            next_char = detokenizer.decode_vector(h_last, temperature=temperature)
            
        current_text += next_char

    return current_text

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
