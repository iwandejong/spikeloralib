import os
import tarfile
import urllib.request
from typing import List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SUBJ_DATASET_URL = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/spikelora/subj")
SUBJECTIVE_FILE = "quote.tok.gt9.5000"  # label 1
OBJECTIVE_FILE = "plot.tok.gt9.5000"  # label 0


def download_subj(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    subj_path = os.path.join(cache_dir, SUBJECTIVE_FILE)
    obj_path = os.path.join(cache_dir, OBJECTIVE_FILE)
    if os.path.exists(subj_path) and os.path.exists(obj_path):
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    archive_path = os.path.join(cache_dir, "rotten_imdb.tar.gz")
    try:
        urllib.request.urlretrieve(SUBJ_DATASET_URL, archive_path)
        with tarfile.open(archive_path) as tar:
            tar.extractall(cache_dir)
    except Exception as exc:  # noqa: BLE001 - re-raise with an actionable message
        raise RuntimeError(
            f"Could not download the subjectivity dataset from {SUBJ_DATASET_URL} ({exc}). "
            f"Download/extract it manually and pass --data_dir pointing at a directory "
            f"containing {SUBJECTIVE_FILE} and {OBJECTIVE_FILE}."
        ) from exc
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)

    if not (os.path.exists(subj_path) and os.path.exists(obj_path)):
        raise RuntimeError(
            f"Downloaded {SUBJ_DATASET_URL} but did not find expected files "
            f"{SUBJECTIVE_FILE}/{OBJECTIVE_FILE} under {cache_dir}."
        )
    return cache_dir


def load_subj(data_dir: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(os.path.join(data_dir, SUBJECTIVE_FILE), encoding="utf-8", errors="ignore") as f:
        for line in f:
            texts.append(line.strip())
            labels.append(1)
    with open(os.path.join(data_dir, OBJECTIVE_FILE), encoding="utf-8", errors="ignore") as f:
        for line in f:
            texts.append(line.strip())
            labels.append(0)
    return texts, labels


def train_val_split(texts: List[str], labels: List[int], test_size: float = 0.1, seed: int = 0):
    return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)


class SubjDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)["input_ids"]
        return ids, self.labels[idx]


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        ids_list, label_list = zip(*batch)
        encoded = tokenizer.pad({"input_ids": list(ids_list)}, padding=True)
        data = torch.tensor(encoded["input_ids"], dtype=torch.long)
        labels = torch.tensor(label_list, dtype=torch.long)
        return data, labels

    return collate_fn
