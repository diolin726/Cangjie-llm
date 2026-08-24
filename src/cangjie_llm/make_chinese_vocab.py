import jieba
import json
import unicodedata
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from .dataset import _load_source_dataset
from .config import dataset_mix
from opencc import OpenCC
TOP_K = 1000
SAMPLE_DOCUMENTS = 10_000
OUTPUT_PATH = Path(__file__).with_name("chinese_vocab_top1000.json")


def main():
    vocab_counter = Counter()
    convertor = OpenCC("s2twp")
    loaded_sources = 0
    total_weight = sum(float(item.get("weight", 1.0)) for item in dataset_mix)
    for mix_item in dataset_mix:
        try:
            ds = _load_source_dataset(dataset_mix=[mix_item], split="train", streaming=True)
        except Exception as error:
            print(f"略過無法載入的資料集 {mix_item['name']}: {error}")
            continue

        try:
            rows_read = 0
            source_limit = max(1, round(SAMPLE_DOCUMENTS * float(mix_item.get("weight", 1.0)) / total_weight))
            for item in tqdm(ds, desc=f"Building vocab: {mix_item['name']}", total=source_limit, disable=True):
                text = unicodedata.normalize("NFKC", item.get("text", "")).replace("\u3000", " ")
                text = convertor.convert(text)
                tokens = (
                    word.strip()
                    for word in jieba.cut(text)
                    if len(word.strip()) >= 2
                    and not word.strip().isnumeric()
                    and not word.strip()[0].isascii()
                )
                vocab_counter.update(tokens)
                rows_read += 1
                if rows_read >= source_limit:
                    break
        except Exception as error:
            print(f"略過無法讀取的資料集 {mix_item['name']}: {error}")
            continue

        if rows_read:
            loaded_sources += 1
            print(f"統計 {mix_item['name']}: {rows_read:,} 篇")

    if not loaded_sources:
        raise RuntimeError("沒有可用的資料集，無法建立中文詞彙檔")

    top_tokens_with_freq = vocab_counter.most_common(TOP_K)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump([word for word, _ in top_tokens_with_freq], file, ensure_ascii=False, indent=2)
    print(f"wrote {len(top_tokens_with_freq)} words to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
