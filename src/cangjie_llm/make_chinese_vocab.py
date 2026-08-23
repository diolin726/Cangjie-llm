import jieba
from .dataset import _load_source_dataset
from .config import dataset_mix
ds=_load_source_dataset(
    json_path=None,
    dataset_name=None,
    dataset_mix=dataset_mix,
    dataset_dir=None,
    data_files=None,
    split="train",
    streaming=False,
    verbose=True,
    )
print( ds[0]["text"] )
