"""Признаки из Vision JSON первого фото. Изображения распаковывать не нужно."""
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd


PHOTO_FEATURES = [
    "photo_metadata_missing", "photo_labels_missing", "photo_colors_missing",
    "photo_label_count", "photo_max_label_score", "photo_cat_score", "photo_dog_score",
    "photo_red", "photo_green", "photo_blue",
]


def features_from_metadata(metadata):
    values = dict.fromkeys(PHOTO_FEATURES, np.nan)
    values.update(photo_metadata_missing=int(metadata is None),
                  photo_labels_missing=1, photo_colors_missing=1)
    if metadata is None:
        return values
    if not isinstance(metadata, dict):
        raise ValueError("Vision JSON должен содержать объект")
    labels = metadata.get("labelAnnotations", [])
    if labels:
        scores = {str(item["description"]).lower(): float(item["score"]) for item in labels}
        values.update(photo_labels_missing=0, photo_label_count=len(labels),
                      photo_max_label_score=max(scores.values()),
                      photo_cat_score=scores.get("cat", 0.0),
                      photo_dog_score=scores.get("dog", 0.0))
    colors = metadata.get("imagePropertiesAnnotation", {}).get("dominantColors", {}).get("colors", [])
    total = sum(float(item.get("pixelFraction", 0)) for item in colors)
    if total > 0:
        values["photo_colors_missing"] = 0
        for channel in ["red", "green", "blue"]:
            # В protobuf JSON нулевой цветовой канал может отсутствовать.
            values[f"photo_{channel}"] = sum(
                float(item.get("pixelFraction", 0)) * float(item["color"].get(channel, 0))
                for item in colors
            ) / total
    return values


def load_photo_features(source, pet_ids, split="train"):
    """Одна строка на PetID, только <PetID>-1.json, без статистик по target."""
    if split not in {"train", "test"}:
        raise ValueError("split должен быть train или test")
    source = Path(source)
    ids = pd.Index(pet_ids)
    if ids.empty or ids.hasnans or not ids.is_unique or (ids.astype(str).str.strip() == "").any():
        raise ValueError("PetID должны быть уникальными и непустыми")
    ids = ids.astype(str)
    if not ids.is_unique:
        raise ValueError("PetID совпадают после преобразования в строки")
    wanted = {f"{pet_id}-1.json": pet_id for pet_id in ids}
    folder = f"{split}_metadata"
    rows = {}

    def read(pet_id, payload, name):
        try:
            rows[pet_id] = features_from_metadata(json.loads(payload))
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError(f"Не удалось разобрать метаданные {name}: {error}") from error

    if source.is_file():
        with source.open("rb") as stream:
            if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Вместо ZIP найден указатель Git LFS. Выполните git lfs pull.")
        with zipfile.ZipFile(source) as archive:
            members = [item for item in archive.infolist()
                       if not item.is_dir() and PurePosixPath(item.filename).parent.name == folder]
            if not members:
                raise FileNotFoundError(f"В архиве нет папки {folder}")
            for member in members:
                filename = PurePosixPath(member.filename).name
                if filename in wanted:
                    pet_id = wanted[filename]
                    if pet_id in rows:
                        raise ValueError(f"Повторный JSON первого фото для {pet_id}")
                    read(pet_id, archive.read(member), member.filename)
    else:
        directory = source / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Нет папки {directory}, укажите ZIP или корень полных данных")
        for filename, pet_id in wanted.items():
            path = directory / filename
            if path.is_file():
                read(pet_id, path.read_bytes(), path)
    missing = features_from_metadata(None)
    result = pd.DataFrame([rows.get(pet_id, missing) for pet_id in ids], columns=PHOTO_FEATURES)
    result.insert(0, "PetID", ids)
    numeric = result[PHOTO_FEATURES].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("В метаданных обнаружены бесконечные значения")
    print(f"{split}: JSON первого фото найден для {len(rows)} из {len(ids)} объявлений", flush=True)
    return result
