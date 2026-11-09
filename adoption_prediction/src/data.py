"""Обзорчик данных"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

TARGET = "AdoptionSpeed"
CLASSES = [0, 1, 2, 3, 4]
NUMERIC = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
CATEGORICAL = [
    "Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
    "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized",
    "Health", "State",
]
REQUIRED = ["PetID", "RescuerID", "Name", "Description"] + NUMERIC + CATEGORICAL


def find_csv(root, filename):
    """Разная локация данных (root/train.csv или root/train/train.csv)."""
    root = Path(root)
    split = "train" if filename == "train.csv" else "test"
    candidates = [root / filename, root / split / filename]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Ожидаю {filename}; пришло {found}. Просмотрено в {candidates}. "
            "Мб не подкючены данные с кагла или нет в data/raw/"
        )
    return found[0]


def discover_root(search_root="/kaggle/input"):
    candidates = sorted({p.parent.parent if p.parent.name == "train" else p.parent
                         for p in Path(search_root).rglob("train.csv")})
    if len(candidates) != 1:
        raise FileNotFoundError(f"Нужно прописать DATA_ROOT; доступные: {candidates}")
    return candidates[0]


def validate_frame(frame, train=True):
    missing = set(REQUIRED + ([TARGET] if train else [])) - set(frame.columns)
    if missing:
        raise ValueError(f"Нет колонки: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Пустой датасет")
    for key in ["PetID", "RescuerID"]:
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Нет ключа {key}")
    if frame.PetID.duplicated().any():
        raise ValueError("Дублилование PetID")
    if train and (frame[TARGET].isna().any() or not frame[TARGET].isin(CLASSES).all()):
        raise ValueError("AdoptionSpeed какой-то другой? Не 0-4")
    for col in NUMERIC + CATEGORICAL:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        invalid = frame[col].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Не числа {col}")
        if np.isinf(numeric.dropna()).any():
            raise ValueError(f"Улетели в бесконечность {col}")
        if col in CATEGORICAL and numeric.dropna().mod(1).ne(0).any():
            raise ValueError(f"Должна быть категория, а не число: {col}")


def load_data(root, with_test=True):
    train = pd.read_csv(find_csv(root, "train.csv"))
    validate_frame(train)
    test = None
    if with_test:
        test = pd.read_csv(find_csv(root, "test.csv"))
        validate_frame(test, train=False)
        if TARGET in test:
            raise ValueError("Тест почему-то размечен?")
        if set(train.PetID) & set(test.PetID):
            raise ValueError("PetID есть и в трейне, и в тесте")
    return train, test


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_description(frame):
    return frame.Description.fillna("").astype(str).str.lower().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
