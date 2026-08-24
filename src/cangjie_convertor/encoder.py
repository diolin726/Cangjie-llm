import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ._shared import load_cj_assets


CHINESE_VOCAB_PATH = Path(__file__).resolve().parent.parent / "cangjie_llm" / "chinese_vocab_top1000.json"


@lru_cache(maxsize=1)
def load_chinese_vocab():
    """Load optional direct Chinese word tokens while preserving base token IDs."""
    if not CHINESE_VOCAB_PATH.exists():
        return ()

    with CHINESE_VOCAB_PATH.open(encoding="utf-8") as file:
        entries = json.load(file)
    if not isinstance(entries, list):
        raise ValueError(f"中文詞彙檔必須是 JSON list: {CHINESE_VOCAB_PATH}")

    seen = set()
    words = []
    for entry in entries:
        if not isinstance(entry, str) or not entry or entry in seen:
            continue
        if entry.startswith("cj_") or entry.startswith("<BYTE_"):
            raise ValueError(f"中文詞彙不可使用保留 token 名稱: {entry}")
        seen.add(entry)
        words.append(entry)
    return tuple(words)


def _vocab_tokens():
    """Return the token order shared by the encoder and model configuration."""
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    cangjie_symbols = [
        'cj_a', 'cj_b', 'cj_c', 'cj_d', 'cj_e', 'cj_f',
        'cj_g', 'cj_h', 'cj_i', 'cj_j', 'cj_k', 'cj_l',
        'cj_m', 'cj_n', 'cj_o', 'cj_p', 'cj_q', 'cj_r',
        'cj_s', 'cj_t', 'cj_u', 'cj_v', 'cj_w', 'cj_x',
        'cj_y', 'cj_z',
    ]
    ascii_chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
    byte_tokens = [f"<BYTE_{i}>" for i in range(256)]
    # Keep the historical 383 entries unchanged so old checkpoints remain reusable.
    return cangjie_symbols + special_tokens + ascii_chars + byte_tokens + list(load_chinese_vocab())


def get_vocab_size():
    """Derive the input vocabulary size from the encoder's canonical token list."""
    return len(_vocab_tokens())


def get_vocab_fingerprint():
    """Fingerprint the direct-token vocabulary for token-cache invalidation."""
    return hashlib.sha256("\n".join(_vocab_tokens()).encode("utf-8")).hexdigest()


class cj_encoder:
    def __init__(self , vocab_size):
        self.vocab_size = vocab_size
        self.make_vocab()
        self.data_map, _, self.reversed_cj_key, _, self.encoded_tokens = load_cj_assets()
        self.pad_id = self.vocab["[PAD]"]
        self.unk_row = (
            self.vocab["[UNK]"],
            self.pad_id,
            self.pad_id,
            self.pad_id,
            self.pad_id,
        )
        self.single_token_rows = {
            token: (token_id, self.pad_id, self.pad_id, self.pad_id, self.pad_id)
            for token, token_id in self.vocab.items()
        }
        self.encoded_token_rows = {
            char: tuple(self.vocab[token] for token in tokens)
            for char, tokens in self.encoded_tokens.items()
        }
        self.chinese_words_by_first_char = {}
        for word in load_chinese_vocab():
            self.chinese_words_by_first_char.setdefault(word[0], []).append(word)
        for words in self.chinese_words_by_first_char.values():
            words.sort(key=lambda word: (-len(word), word))
        self.byte_rows_cache = {}
        self.char_to_output_id = {
            char: self.reversed_cj_key[key] + self.vocab_size - 26
            for char, key in self.data_map.items()
        }

    def make_vocab(self):
        self.vocab={}
        self.id_to_vocab=[]
        for token in _vocab_tokens():
            self.vocab[token] = len(self.vocab)
            self.id_to_vocab.append(token)

    def encode(self, s):
        encoded_tokens = self.encoded_tokens
        return [encoded_tokens.get(token, [token]) for token in s]

    def encode_text_to_rows(self, s):
        encoded_token_rows = self.encoded_token_rows
        single_token_rows = self.single_token_rows
        byte_rows_cache = self.byte_rows_cache
        unk_row = self.unk_row
        pad_id = self.pad_id
        vocab = self.vocab
        rows = []
        append = rows.append
        extend = rows.extend

        index = 0
        while index < len(s):
            token = s[index]
            matched_word = next(
                (
                    word
                    for word in self.chinese_words_by_first_char.get(token, ())
                    if s.startswith(word, index)
                ),
                None,
            )
            if matched_word is not None:
                append(single_token_rows[matched_word])
                index += len(matched_word)
                continue

            encoded = encoded_token_rows.get(token)
            if encoded is not None:
                append(encoded)
                index += 1
                continue

            single = single_token_rows.get(token)
            if single is not None:
                append(single)
                index += 1
                continue

            cached = byte_rows_cache.get(token)
            if cached is None:
                try:
                    cached = tuple(
                        (vocab[f"<BYTE_{byte}>"], pad_id, pad_id, pad_id, pad_id)
                        for byte in token.encode("utf-8")
                    )
                except Exception:
                    cached = (unk_row,)
                byte_rows_cache[token] = cached
            extend(cached)
            index += 1

        return rows

    def encode_to_id(self , text): # Warning!! only when input are in token table
        char_to_output_id = self.char_to_output_id
        vocab = self.vocab
        return [char_to_output_id.get(token, vocab[token] - 26) for token in text]

if __name__ == '__main__':
    e = cj_encoder()
    a = e.encode("我是abc123🥰")
    print(a)
    print(len(e.data_map))
