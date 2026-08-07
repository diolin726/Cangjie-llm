import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# 從原 pre-train-llama 模型檔載入底層模組
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pre-train-llama'))
from modeling_llama_custom import DecoderLayer, RotaryPositionalEncoding, create_causal_mask


BASE_VOCAB_SIZE = 10000
NUM_CJ_TOKENS = 27
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + NUM_CJ_TOKENS  # 10027

class CangjieCompositeEmbedding(nn.Module):
    """
    倉頡 768 維位置線性變換融合 Embedding 模組：
    - 非中文 Token：查原有 Embedding 表 E(token_id)
    - 中文 Token：拆碼成 5 個倉頡字根 Token，經由 5 個獨立的 768 維位置變換矩陣 W_1...W_5 投影後加總：
      E_中文 = W_1(E(c_1)) + W_2(E(c_2)) + W_3(E(c_3)) + W_4(E(c_4)) + W_5(E(c_5))
    """
    def __init__(self, vocab_size=TOTAL_VOCAB_SIZE, hidden_dim=768, num_positions=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_positions = num_positions
        
        # 共享 Embedding 矩陣 (包含 10,000 原 token + 27 個倉頡 token)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 5 個獨立的位置 768 維線性變換矩陣 W_1 ... W_5 (768 -> 768)
        self.pos_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_positions)
        ])
        
        # 初始化位置投影矩陣為單位矩陣 (Identity Matrix)，確保初期特徵平滑過渡
        for proj in self.pos_projections:
            nn.init.eye_(proj.weight)

    def forward(self, standard_ids, is_chinese_mask, cangjie_ids):
        """
        standard_ids: (B, T) LongTensor
        is_chinese_mask: (B, T) BoolTensor
        cangjie_ids: (B, T, 5) LongTensor
        """
        B, T = standard_ids.shape
        
        # 1. 計算非中文 Token 的標準 Embedding (B, T, 768)
        std_emb = self.embedding(standard_ids)
        
        # 2. 查表獲取 5 個倉頡字根 Token 的向量 (B, T, 5, 768)
        cj_embs = self.embedding(cangjie_ids)
        
        # 3. 依據位置 i 分別進行 768 維線性投影 W_i(E(c_i)) 並求和 (B, T, 768)
        cj_composite_emb = torch.zeros(B, T, self.hidden_dim, device=standard_ids.device, dtype=std_emb.dtype)
        for i in range(self.num_positions):
            pos_emb_i = cj_embs[:, :, i, :]  # (B, T, 768)
            proj_i = self.pos_projections[i](pos_emb_i)  # (B, T, 768)
            cj_composite_emb = cj_composite_emb + proj_i
            
        # 4. 使用 is_chinese_mask 融合：中文用 cj_composite_emb，非中文用 std_emb
        final_emb = torch.where(is_chinese_mask.unsqueeze(-1), cj_composite_emb, std_emb)
        return final_emb

class CangjieTextGenerationModel(nn.Module):
    """
    整合倉頡組合 Embedding 的 Llama 樣式 Decoder 語言模型
    """
    def __init__(self, num_layers=8, num_heads=8, num_kv_heads=4, hidden_dim=768,
                 max_seq_len=512, vocab_size=TOTAL_VOCAB_SIZE, dropout=0.1):
        super().__init__()
        self.config = {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "hidden_dim": hidden_dim,
            "max_seq_len": max_seq_len,
            "vocab_size": vocab_size,
            "dropout": dropout
        }
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = CangjieCompositeEmbedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, standard_ids, is_chinese_mask, cangjie_ids, mask=None):
        # 1. 取得組合 Embedding 向量
        x = self.embedding(standard_ids, is_chinese_mask, cangjie_ids)
        
        # 2. 建立預設因果遮罩 (Causal Mask)
        if mask is None:
            seq_len = standard_ids.shape[1]
            mask = create_causal_mask(seq_len, standard_ids.device)
            
        # 3. 經過 Decoder Layers
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
            
        # 4. RMSNorm 與 LM Head
        x = self.norm(x)
        logits = self.out(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_dir='pre-train-llama'):
        """
        從預訓練權重加載模型，並無縫升級輸入/輸出層以支援倉頡 Token
        """
        config_path = os.path.join(model_dir, 'config.json')
        weights_path = os.path.join(model_dir, 'model.safetensors')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)["model_config"]
            
        model = cls(
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            hidden_dim=cfg["hidden_dim"],
            max_seq_len=cfg["max_seq_len"],
            vocab_size=TOTAL_VOCAB_SIZE,
            dropout=cfg.get("dropout", 0.1)
        )
        
        pretrained_weights = load_file(weights_path)
        
        # 複製前 10,000 個原始 Token 的 Embedding
        with torch.no_grad():
            old_emb = pretrained_weights["embedding.weight"]
            model.embedding.embedding.weight[:BASE_VOCAB_SIZE].copy_(old_emb)
            
            # 初始化新增的 27 個倉頡 Token 的 Embedding (取原始均值與方差)
            mean_emb = old_emb.mean(dim=0)
            std_emb = old_emb.std(dim=0)
            for i in range(NUM_CJ_TOKENS):
                model.embedding.embedding.weight[BASE_VOCAB_SIZE + i].copy_(
                    mean_emb + torch.randn_like(mean_emb) * 0.02
                )
                
            # 複製 Decoder 各層與 RMSNorm 權重
            for name, param in model.named_parameters():
                if name.startswith("decoders.") or name.startswith("norm."):
                    if name in pretrained_weights:
                        param.copy_(pretrained_weights[name])
                        
            # 複製前 10,000 個原始 Token 的 LM Head 權重
            old_out_w = pretrained_weights["out.weight"]
            model.out.weight[:BASE_VOCAB_SIZE].copy_(old_out_w)
            if "out.bias" in pretrained_weights and pretrained_weights["out.bias"] is not None:
                model.out.bias[:BASE_VOCAB_SIZE].copy_(pretrained_weights["out.bias"])

        print(f"成功加載預訓練模型 '{model_dir}'，並擴充詞表至 {TOTAL_VOCAB_SIZE}。")
        return model

if __name__ == '__main__':
    model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
    print("模型加載成功！總參數量:", sum(p.numel() for p in model.parameters()))
    print("W_1 ... W_5 投影矩陣數量:", len(model.embedding.pos_projections))
