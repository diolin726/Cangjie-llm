import jieba
import json
import unicodedata
from collections import Counter
from tqdm import tqdm
from .dataset import _load_source_dataset
from .config import dataset_mix

ds=_load_source_dataset(
    json_path=None,
    dataset_name=None,
    dataset_mix=dataset_mix,
    dataset_dir=None,
    data_files=None,
    split="train",
    streaming=True,
    verbose=True,
    )

vocab_counter = Counter()
TOP_K = 1000

for _ , item in enumerate(tqdm(ds, desc="Building vocab")):
    text = item.get("text", "")
    text = unicodedata.normalize("NFKC", text).replace("\u3000", " ")
    tokens = (word.strip() for word in jieba.cut(text) if len(word.strip()) >= 2 and not word.strip().isnumeric())
    vocab_counter.update(tokens)

top_tokens_with_freq = vocab_counter.most_common(TOP_K)
top_tokens = [word for word, count in top_tokens_with_freq]

print("\n--- Top 20 Tokens ---")
for word, count in top_tokens_with_freq[:20]:
    print(f"{word}: {count:,}")

with open("chinese_vocab_top1000.json", "w", encoding="utf-8") as f:
    json.dump(top_tokens, f, ensure_ascii=False, indent=2)
