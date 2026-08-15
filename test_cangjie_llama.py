import unittest
import torch
import torch.nn as nn
from cangjie_converter import ALL_CJ_TOKENS, PAD_CJ_TOKEN, get_char_cangjie_tokens
from cangjie_tokenizer import CangjieLlamaTokenizer, CJ_TOKEN_TO_ID, PAD_CJ_ID
from modeling_cangjie_llama import CangjieTextGenerationModel, CangjieCompositeEmbedding, TOTAL_VOCAB_SIZE

class TestCangjieLlamaSystem(unittest.TestCase):

    def test_01_cangjie_converter(self):
        """測試 1: 倉頡轉換器與 27 個 Token 拆碼補齊"""
        self.assertEqual(len(ALL_CJ_TOKENS), 27, "倉頡 Token 總數必須剛好為 27 (26 字母 + 1 [PAD_CJ])")
        self.assertIn(PAD_CJ_TOKEN, ALL_CJ_TOKENS, "[PAD_CJ] 必須在 Token 列表中")
        
        ming_tokens = get_char_cangjie_tokens('明', max_len=5)
        self.assertEqual(ming_tokens, ['日', '月', '[PAD_CJ]', '[PAD_CJ]', '[PAD_CJ]'])
        
        jing_tokens = get_char_cangjie_tokens('晶', max_len=5)
        self.assertEqual(jing_tokens, ['日', '日', '日', '[PAD_CJ]', '[PAD_CJ]'])

    def test_02_cangjie_tokenizer(self):
        """測試 2: 分詞器雙軌 Encoding (標準 BPE + 倉頡 5 碼)"""
        tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
        text = "Hello 明林 World!"
        std_ids, mask, cj_ids = tokenizer.encode(text)
        
        self.assertEqual(std_ids.dim(), 1)
        self.assertEqual(mask.dim(), 1)
        self.assertEqual(cj_ids.dim(), 2)
        self.assertEqual(cj_ids.shape[1], 5, "每個 Token 的倉頡 ID 向量第二維度必須為 5")
        
        # 驗證 '明' (索引 4) 與 '林' (索引 5)
        # 找到被標記為中文的 token index
        cn_indices = [i for i, v in enumerate(mask.tolist()) if v]
        self.assertGreaterEqual(len(cn_indices), 2)
        # 驗證前兩個中文 token 的倉頡拆碼
        first_cn_idx = cn_indices[0]
        second_cn_idx = cn_indices[1]
        ming_radicals = [ALL_CJ_TOKENS[i - 10000] for i in cj_ids[first_cn_idx].tolist()]
        self.assertEqual(ming_radicals, ['日', '月', '[PAD_CJ]', '[PAD_CJ]', '[PAD_CJ]'])

    def test_03_composite_embedding_gradients(self):
        """測試 3: 768 維位置變換矩陣 W_1...W_5 與 倉頡 Embedding 的梯度流向"""
        emb_layer = CangjieCompositeEmbedding(vocab_size=TOTAL_VOCAB_SIZE, hidden_dim=768)
        
        tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
        std_ids, mask, cj_ids = tokenizer.encode("測試明晶")
        
        std_ids = std_ids.unsqueeze(0)  # (1, T)
        mask = mask.unsqueeze(0)        # (1, T)
        cj_ids = cj_ids.unsqueeze(0)    # (1, T, 5)
        
        # Forward Pass
        out_emb = emb_layer(std_ids, mask, cj_ids)
        # 檢查輸出 shape 與輸入 token 數相容
        self.assertEqual(out_emb.shape, (1, std_ids.shape[1], 768))
        
        # Backward Pass 計算梯度
        loss = out_emb.sum()
        loss.backward()
        
        # 驗證 5 個 W_1 ... W_5 投影矩陣的梯度皆不為 None 且非全零
        for idx, proj in enumerate(emb_layer.pos_projections):
            self.assertIsNotNone(proj.weight.grad, f"W_{idx+1} 投影矩陣梯度不應為 None")
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0, f"W_{idx+1} 投影矩陣梯度應大於 0")
            
        # 驗證倉頡 Token 的 Embedding 梯度 (拆為 cj_embedding)
        cj_grad = emb_layer.cj_embedding.weight.grad
        self.assertIsNotNone(cj_grad)
        # 驗證 27 個倉頡字根 embedding 有接收到梯度
        self.assertGreater(cj_grad.abs().sum().item(), 0)

    def test_04_full_model_forward(self):
        """測試 4: 完整 Llama 模型 Forward Pass 與 Logits 輸出尺寸"""
        model = CangjieTextGenerationModel.from_pretrained('pre-train-llama')
        model.eval()
        
        tokenizer = CangjieLlamaTokenizer('pre-train-llama/tokenizer.json')
        std_ids, mask, cj_ids = tokenizer.encode("Hello 倉頡!")
        
        std_ids = std_ids.unsqueeze(0)
        mask = mask.unsqueeze(0)
        cj_ids = cj_ids.unsqueeze(0)
        
        with torch.no_grad():
            logits = model(std_ids, mask, cj_ids)
            
        self.assertEqual(logits.shape, (1, std_ids.shape[1], TOTAL_VOCAB_SIZE))
        print("全模型 Forward Pass 測試成功，Logits 尺寸:", logits.shape)

if __name__ == '__main__':
    unittest.main()
