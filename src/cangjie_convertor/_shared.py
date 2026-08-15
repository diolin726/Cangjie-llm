import json
from functools import lru_cache
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "cj5.json"


@lru_cache(maxsize=1)
def load_cj_assets():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_map = json.load(f)

    cj_keys = []
    cj_decodemap = {}
    encoded_tokens = {}

    for char, key in data_map.items():
        cj_keys.append(key)
        if key not in cj_decodemap:
            cj_decodemap[key] = []
        cj_decodemap[key].append(char)

        padded_tokens = [f"cj_{token}" for token in key]
        padded_tokens.extend(["[PAD]"] * (5 - len(padded_tokens)))
        encoded_tokens[char] = tuple(padded_tokens)

    unique_cj_keys = list(set(cj_keys))
    reversed_cj_key = {key: i for i, key in enumerate(unique_cj_keys)}
    return data_map, unique_cj_keys, reversed_cj_key, cj_decodemap, encoded_tokens

