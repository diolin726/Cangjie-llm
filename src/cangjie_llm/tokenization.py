import unicodedata
from functools import lru_cache

import torch

from cangjie_convertor import cj_decoder, cj_encoder

from .config import (
    common_char_table_size,
    common_word_max_length,
    common_word_table_size,
    detokenize_beam_size,
    detokenize_context_window,
    vocab_size,
)


def _is_cjk_char(char):
    if not isinstance(char, str) or len(char) != 1:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _is_basic_cjk_char(char):
    return isinstance(char, str) and len(char) == 1 and 0x4E00 <= ord(char) <= 0x9FFF


def _is_cjk_word(word):
    return (
        isinstance(word, str)
        and 2 <= len(word) <= common_word_max_length
        and all(_is_cjk_char(char) for char in word)
    )


@lru_cache(maxsize=1)
def get_common_char_rank():
    import jieba

    jieba.initialize()
    single_char_freq = [
        (char, freq)
        for char, freq in jieba.dt.FREQ.items()
        if _is_cjk_char(char) and isinstance(freq, int) and freq > 0
    ]
    single_char_freq.sort(key=lambda item: (-item[1], item[0]))
    return {
        char: rank
        for rank, (char, _) in enumerate(single_char_freq[:common_char_table_size])
    }


@lru_cache(maxsize=1)
def get_common_words_by_last_char():
    import jieba

    jieba.initialize()
    common_words = [
        (word, freq)
        for word, freq in jieba.dt.FREQ.items()
        if _is_cjk_word(word) and isinstance(freq, int) and freq > 0
    ]
    common_words.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))

    words_by_last_char = {}
    for rank, (word, freq) in enumerate(common_words[:common_word_table_size]):
        words_by_last_char.setdefault(word[-1], []).append((word, rank, freq))

    return {
        char: tuple(entries)
        for char, entries in words_by_last_char.items()
    }


def _best_common_word_score(context, candidate, common_words_by_last_char):
    best_score = (-1, -1, -common_word_table_size)
    for word, rank, freq in common_words_by_last_char.get(candidate, ()):
        prefix = word[:-1]
        if context.endswith(prefix):
            score = (len(word), freq, -rank)
            if score > best_score:
                best_score = score
    return best_score


def _best_context_word_freq(context, candidate, freq_table):
    best_freq = 0
    for start in range(len(context) + 1):
        word = context[start:] + candidate
        freq = freq_table.get(word, 0)
        if freq > best_freq:
            best_freq = freq
    return best_freq


def _candidate_local_score(
    context,
    candidate,
    common_words_by_last_char,
    freq_table,
    common_char_rank,
):
    common_word_score = _best_common_word_score(context, candidate, common_words_by_last_char)
    if common_word_score[0] < 0:
        common_word_score = (0, 0, 0)
    return (
        common_word_score[0],
        common_word_score[1],
        common_word_score[2],
        _best_context_word_freq(context, candidate, freq_table),
        int(candidate in common_char_rank),
        freq_table.get(candidate, 0),
        int(_is_basic_cjk_char(candidate)),
        -common_char_rank.get(candidate, common_char_table_size),
    )


def _add_score_tuple(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _context_from_tokens(tokens, limit):
    chars = [
        token for token in tokens
        if isinstance(token, str) and len(token) == 1
    ]
    if limit <= 0:
        return "".join(chars)
    return "".join(chars[-limit:])


class tokenizer:
    def __init__(self):
        self.decoder = cj_decoder()
        self.cj_encoder = cj_encoder(vocab_size)
        self.vocab = self.cj_encoder.vocab
        self.id_to_vocab = self.cj_encoder.id_to_vocab

    def all_vocab(self):
        """建立所有 13228 個輸出詞彙的 5-tuple tensor"""
        pad_id = self.vocab["[PAD]"]
        codes = []
        self.output_tokens = []

        # 1) 非 CJK tokens (357個) #383 - 26 = 357
        for token_name, token_id in self.vocab.items():
            if not token_name.startswith("cj_"):
                codes.append([token_id, pad_id, pad_id, pad_id, pad_id])
                self.output_tokens.append(token_name)

        # 2) 倉頡排列 (12871個)
        for code in self.decoder.cj_keys:
            ids = [self.vocab[f"cj_{char}"] for char in code]
            ids += [pad_id] * (5 - len(ids))
            codes.append(ids)
            self.output_tokens.append(code)

        return torch.tensor(codes, dtype=torch.long)

    def tokenlist_to_id(self, token_list):
        if len(token_list) == 5:
            return [[self.vocab[token] for token in token_list]]
        if token_list[0] in self.vocab:
            pad_id = self.vocab["[PAD]"]
            return [[self.vocab[token_list[0]], pad_id, pad_id, pad_id, pad_id]]

        try:
            utf8_bytes = token_list[0].encode("utf-8")
            pad_id = self.vocab["[PAD]"]
            return [
                [self.vocab[f"<BYTE_{byte}>"], pad_id, pad_id, pad_id, pad_id]
                for byte in utf8_bytes
            ]
        except Exception:
            pad_id = self.vocab["[PAD]"]
            return [[self.vocab["[UNK]"], pad_id, pad_id, pad_id, pad_id]]

    def tokenize(self, text):
        normalized = unicodedata.normalize("NFKC", text).replace("\u3000", " ")
        return self.cj_encoder.encode_text_to_rows(normalized)

    def id_decode(self, ids):
        ans = []
        for token_id in ids:
            if token_id < vocab_size - 26:
                ans.append(self.id_to_vocab[token_id + 26])
            else:
                ans.append(self.decoder.id_decode(token_id + 26 - vocab_size))
        return ans

    def detokenize(self, id_list):
        import jieba

        jieba.initialize()
        common_char_rank = get_common_char_rank()
        common_words_by_last_char = get_common_words_by_last_char()
        freq_table = jieba.dt.FREQ

        raw = self.id_decode(id_list)

        # 合併連續的 <BYTE_x> token，還原成原本的 utf-8 字元（例如 emoji）
        merged = []
        idx = 0
        while idx < len(raw):
            item = raw[idx]
            if isinstance(item, str) and item.startswith("<BYTE_"):
                byte_buf = []
                while idx < len(raw) and isinstance(raw[idx], str) and raw[idx].startswith("<BYTE_"):
                    byte_buf.append(int(raw[idx][6:-1]))
                    idx += 1
                try:
                    merged.append(bytes(byte_buf).decode("utf-8"))
                except UnicodeDecodeError:
                    merged.append("�")
                continue
            merged.append(item)
            idx += 1

        # 用 beam search 消歧倉頡同碼字，避免逐字 greedy 太早定案。
        beams = [((0, 0, 0, 0, 0, 0, 0, 0), [])]
        for item in merged:
            if not isinstance(item, list):
                for beam_idx, (score, tokens) in enumerate(beams):
                    next_tokens = list(tokens)
                    next_tokens.append(item)
                    beams[beam_idx] = (score, next_tokens)
                continue

            candidates = list(dict.fromkeys(
                cand for cand in item
                if isinstance(cand, str) and len(cand) == 1
            ))
            if not candidates:
                fallback = item[0] if item else ""
                for beam_idx, (score, tokens) in enumerate(beams):
                    next_tokens = list(tokens)
                    next_tokens.append(fallback)
                    beams[beam_idx] = (score, next_tokens)
                continue

            expanded_beams = []
            for score, tokens in beams:
                context = _context_from_tokens(tokens, detokenize_context_window)
                for cand in candidates:
                    local_score = _candidate_local_score(
                        context,
                        cand,
                        common_words_by_last_char,
                        freq_table,
                        common_char_rank,
                    )
                    next_tokens = list(tokens)
                    next_tokens.append(cand)
                    expanded_beams.append((_add_score_tuple(score, local_score), next_tokens))

            deduped_beams = {}
            for score, tokens in expanded_beams:
                text_key = "".join(tokens)
                prev = deduped_beams.get(text_key)
                if prev is None or score > prev[0]:
                    deduped_beams[text_key] = (score, tokens)

            beams = sorted(
                deduped_beams.values(),
                key=lambda beam: beam[0],
                reverse=True,
            )[:detokenize_beam_size]

        return "".join(beams[0][1]) if beams else ""


__all__ = ["tokenizer"]
