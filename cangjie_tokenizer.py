import torch
from tokenizers import Tokenizer
from cangjie_converter import ALL_CJ_TOKENS, PAD_CJ_TOKEN, get_char_cangjie_tokens

# 27 個倉頡 Token 的 ID 映射 (10000 ~ 10026)
BASE_VOCAB_SIZE = 10000
CJ_TOKEN_TO_ID = {tok: BASE_VOCAB_SIZE + i for i, tok in enumerate(ALL_CJ_TOKENS)}
ID_TO_CJ_TOKEN = {v: k for k, v in CJ_TOKEN_TO_ID.items()}
PAD_CJ_ID = CJ_TOKEN_TO_ID[PAD_CJ_TOKEN]
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + len(ALL_CJ_TOKENS)  # 10027

def is_chinese_char(char: str) -> bool:
    """判斷單個字符是否為 CJK 中文字符"""
    if len(char) != 1:
        return False
    cp = ord(char)
    return (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF)

class CangjieLlamaTokenizer:
    def __init__(self, tokenizer_json_path: str = 'pre-train-llama/tokenizer.json'):
        self.base_tokenizer = Tokenizer.from_file(tokenizer_json_path)
        self.cj_token_to_id = CJ_TOKEN_TO_ID
        self.pad_cj_id = PAD_CJ_ID
        self.total_vocab_size = TOTAL_VOCAB_SIZE

    def encode(self, text: str, max_seq_len: int = 512, add_eos: bool = True):
        """
        將輸入字串轉為三大 Tensor：
        1. standard_ids: (T,) - 原有 BPE 詞庫的 ID (0~9999)
        2. is_chinese_mask: (T,) - bool，標示該 token 是否為中文文字
        3. cangjie_ids: (T, 5) - 該 token 對應的 5 個倉頡字根 ID (10000~10026)
        
        :param add_eos: 若為 True，自動在序列末尾添加 [eos] 結束標籤 (Token ID: 1)
        """
        standard_ids = []
        is_chinese_mask = []
        cangjie_ids = []

        # 逐字解析 text，以兼顧中文文字與原有英文/符號
        i = 0
        n = len(text)
        limit = max_seq_len - 1 if add_eos else max_seq_len

        while i < n and len(standard_ids) < limit:
            char = text[i]
            if is_chinese_char(char):
                # 中文字符：標記為中文，並獲取 5 個倉頡字根 ID
                standard_ids.append(0)  # 使用 0 ([pad]) 作為佔位符
                is_chinese_mask.append(True)
                
                cj_radicals = get_char_cangjie_tokens(char, max_len=5)
                cj_ids = [CJ_TOKEN_TO_ID[r] for r in cj_radicals]
                cangjie_ids.append(cj_ids)
                i += 1
            else:
                # 非中文子字串（英文、符號等）：積攢非中文字段進行 base tokenization
                start = i
                while i < n and not is_chinese_char(text[i]):
                    i += 1
                non_chinese_segment = text[start:i]
                
                encoded_seg = self.base_tokenizer.encode(non_chinese_segment)
                for token_id in encoded_seg.ids:
                    if len(standard_ids) >= limit:
                        break
                    standard_ids.append(token_id)
                    is_chinese_mask.append(False)
                    cangjie_ids.append([PAD_CJ_ID] * 5)

        # 自動在資料末尾添加 [eos] (Token ID: 1)
        if add_eos and len(standard_ids) < max_seq_len:
            standard_ids.append(1)  # [eos] Token ID
            is_chinese_mask.append(False)
            cangjie_ids.append([PAD_CJ_ID] * 5)

        # 轉為 PyTorch Tensors
        standard_ids_t = torch.tensor(standard_ids, dtype=torch.long)
        is_chinese_mask_t = torch.tensor(is_chinese_mask, dtype=torch.bool)
        cangjie_ids_t = torch.tensor(cangjie_ids, dtype=torch.long)

        return standard_ids_t, is_chinese_mask_t, cangjie_ids_t


if __name__ == '__main__':
    tok = CangjieLlamaTokenizer()
    sample = "Hello 明林 World!"
    std_ids, mask, cj_ids = tok.encode(sample)
    print("Sample:", sample)
    print("Standard IDs shape:", std_ids.shape, std_ids)
    print("Chinese mask:", mask)
    print("Cangjie IDs shape:", cj_ids.shape)
    print("Cangjie IDs for Chinese tokens:")
    for idx, is_cn in enumerate(mask):
        if is_cn:
            radicals = [ID_TO_CJ_TOKEN[id_val.item()] for id_val in cj_ids[idx]]
            print(f"  Token index {idx}: {radicals}")
