import torch
import torch.nn.functional as F
from cangjie_converter import load_cangjie_dict, get_char_cangjie_tokens
from cangjie_tokenizer import CJ_TOKEN_TO_ID, BASE_VOCAB_SIZE

class CangjieCosineDetokenizer:
    """
    構想 C：向量餘弦相似度匹配 Detokenizer (Cosine Similarity Retrieval Detokenizer)
    
    原理：
    1. 將字庫中收錄的 65,176 個中文漢字，透過模型當前的 W_1...W_5 矩陣與 Embedding 算出每個字的 768 維組合向量。
    2. 構建 65,176 x 768 的 Codebook 矩陣並進行 L2 Normalize。
    3. 當 LLM 輸出最後一層 768 維向量 H_last 時，計算 H_last 與 Codebook 的餘弦相似度：
       Similarity = Normalize(H_last) @ Codebook_Normalized.T
    4. 取相似度最大（或透過 Temperature 採樣）的漢字作為解碼結果。
    """
    def __init__(self, model=None):
        self.cj_dict = load_cangjie_dict()
        self.char_list = list(self.cj_dict.keys())
        self.num_chars = len(self.char_list)
        self.codebook_norm = None
        self.device = 'cpu'
        
        if model is not None:
            self.build_codebook(model)

    def build_codebook(self, model, device=None):
        """建構包含 65,176 個漢字的 768 維 L2 Normalized Codebook 矩陣"""
        if device is None:
            device = next(model.parameters()).device
        self.device = device

        cache_path = os.path.join(os.path.dirname(__file__), '.codebook_cache.pt')
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location=device)
                self.codebook_norm = data['codebook_norm']
                self.char_list = data.get('char_list', self.char_list)
                print("從快取載入 codebook 矩陣。")
                return
            except Exception:
                pass

        print(f"正在建構 65,176 個漢字的 768 維 Codebook 矩陣 (裝置: {device})...")
        
        cj_ids_list = []
        for c in self.char_list:
            radicals = get_char_cangjie_tokens(c, max_len=5)
            cj_ids_list.append([CJ_TOKEN_TO_ID[r] for r in radicals])
            
        cj_ids_t = torch.tensor(cj_ids_list, dtype=torch.long, device=device)  # (65176, 5)

        # 轉為相對索引 0..26
        cj_indices = cj_ids_t - BASE_VOCAB_SIZE

        with torch.no_grad():
            cj_embs = model.embedding.cj_embedding(cj_indices)  # (65176, 5, 768)
            codebook_embs = torch.zeros(self.num_chars, model.config["hidden_dim"], device=device)
            for i in range(5):
                codebook_embs += model.embedding.pos_projections[i](cj_embs[:, i, :])
                
        # L2 正規化 (L2 Normalization)
        self.codebook_norm = F.normalize(codebook_embs, p=2, dim=-1)  # (65176, 768)
        try:
            torch.save({'codebook_norm': self.codebook_norm, 'char_list': self.char_list}, cache_path)
            print(f"Codebook 建構完成並快取至 {cache_path} (矩陣形狀: {self.codebook_norm.shape})")
        except Exception:
            print("Codebook 建構完成，但無法寫入快取檔案。")

    def decode_vector(self, hidden_vector: torch.Tensor, top_k: int = 1, temperature: float = 1.0):
        """
        將輸入的 768 維隱層向量 H_last 反向解碼為最匹配的漢字
        
        hidden_vector: (768,) 或 (B, 768)
        """
        if self.codebook_norm is None:
            raise RuntimeError("Codebook 尚未建構，請先呼叫 build_codebook(model)！")

        if hidden_vector.dim() == 1:
            hidden_vector = hidden_vector.unsqueeze(0)  # (1, 768)
            
        # 1. 向量 L2 正規化
        h_norm = F.normalize(hidden_vector.to(self.device), p=2, dim=-1)  # (B, 768)
        
        # 2. 計算餘弦相似度 (B, 65176)
        similarities = torch.matmul(h_norm, self.codebook_norm.T)
        
        # 3. 選取最高相似度的漢字
        results = []
        if temperature <= 0:
            top_indices = torch.argmax(similarities, dim=-1)
            for idx in top_indices.tolist():
                results.append(self.char_list[idx])
        else:
            # 帶有溫度的 Softmax 採樣
            logits = similarities / temperature
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=top_k)
            for batch_indices in sampled_indices.tolist():
                chars = [self.char_list[idx] for idx in batch_indices]
                results.append(chars if top_k > 1 else chars[0])
                
        return results if len(results) > 1 else results[0]

if __name__ == '__main__':
    from modeling_cangjie_llama import CangjieTextGenerationModel
    
    print("測試 768 維餘弦相似度 Detokenizer 運作...")
    model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
    detok = CangjieCosineDetokenizer(model)
    
    # 模擬輸入 "晶" 的 768 維向量
    jing_radicals = get_char_cangjie_tokens('晶', max_len=5)
    jing_ids = torch.tensor([[CJ_TOKEN_TO_ID[r] for r in jing_radicals]])
    jing_indices = jing_ids - BASE_VOCAB_SIZE

    with torch.no_grad():
        jing_embs = model.embedding.cj_embedding(jing_indices)
        simulated_vec = torch.zeros(1, 768)
        for i in range(5):
            simulated_vec += model.embedding.pos_projections[i](jing_embs[:, i, :])
            
    matched_char = detok.decode_vector(simulated_vec, temperature=0.001 , top_k = 5)
    print(f"模擬 '晶' 的 768 維向量解碼結果: {matched_char}")
