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
        try:
            self.base_tokenizer = Tokenizer.from_file(tokenizer_json_path)
        except Exception:
            # fallback simple char-level tokenizer for offline testing
            class SimpleBaseTokenizer:
                def encode(self, text):
                    # return a simple object with .ids as a list of ints
                    ids = [ord(c) % BASE_VOCAB_SIZE for c in text]
                    return type('Enc', (), {'ids': ids})
                def decode(self, ids):
                    try:
                        return ''.join(chr(i) for i in ids)
                    except Exception:
                        return ''
            self.base_tokenizer = SimpleBaseTokenizer()
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

    # --- Incremental / cached encoding helpers ---
    class TokenizerCache:
        """A small helper to cache encoded tensors and append tokens incrementally."""
        def __init__(self, tokenizer: 'CangjieLlamaTokenizer', max_seq_len: int = 512, add_eos: bool = True):
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len
            self.add_eos = add_eos
            self.std_ids = []
            self.is_chinese = []
            self.cj_ids = []

        @classmethod
        def from_text(cls, tokenizer: 'CangjieLlamaTokenizer', text: str, max_seq_len: int = 512, add_eos: bool = True):
            inst = cls(tokenizer, max_seq_len, add_eos)
            std, mask, cj = tokenizer.encode(text, max_seq_len=max_seq_len, add_eos=add_eos)
            inst.std_ids = std.tolist()
            inst.is_chinese = mask.tolist()
            inst.cj_ids = [row.tolist() for row in cj.tolist()]
            return inst

        def to_tensors(self):
            import torch
            return (torch.tensor(self.std_ids, dtype=torch.long),
                    torch.tensor(self.is_chinese, dtype=torch.bool),
                    torch.tensor(self.cj_ids, dtype=torch.long))

        def append_bpe_id(self, bpe_id: int):
            # append a BPE id (non-Chinese)
            if len(self.std_ids) >= self.max_seq_len:
                raise IndexError("tokenizer cache reached max_seq_len")
            self.std_ids.append(int(bpe_id))
            self.is_chinese.append(False)
            self.cj_ids.append([self.tokenizer.pad_cj_id] * 5)

        def append_chinese_char(self, char: str):
            # append a chinese char by converting to cangjie ids
            if len(self.std_ids) >= self.max_seq_len:
                raise IndexError("tokenizer cache reached max_seq_len")
            self.std_ids.append(0)
            self.is_chinese.append(True)
            radicals = get_char_cangjie_tokens(char, max_len=5)
            cj_ids = [self.tokenizer.cj_token_to_id[r] for r in radicals]
            self.cj_ids.append(cj_ids)

        def append_eos(self):
            if self.add_eos and len(self.std_ids) < self.max_seq_len:
                self.append_bpe_id(1)


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
