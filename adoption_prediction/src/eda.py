"""Проверка данных"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CATEGORICAL, CLASSES, NUMERIC, REQUIRED, TARGET, find_csv, normalized_description
from .evaluate import save_json

CODE_RANGES = {"Type": [1, 2], "Gender": [1, 2, 3],
               "MaturitySize": [0, 1, 2, 3, 4], "FurLength": [0, 1, 2, 3],
               "Health": [0, 1, 2, 3], "Vaccinated": [1, 2, 3],
               "Dewormed": [1, 2, 3], "Sterilized": [1, 2, 3]}


def frame_audit(frame, labeled):
    missing_columns = sorted(set(REQUIRED + ([TARGET] if labeled else [])) - set(frame))
    description = normalized_description(frame) if "Description" in frame else pd.Series(dtype=str)
    nonempty = description[description.ne("")]
    result = {
        "shape": list(frame.shape), "missing_columns": missing_columns,
        "null_counts": {col: int(count) for col, count in frame.isna().sum().items()},
        "empty_string_counts": {col: int(frame[col].fillna("").astype(str).str.strip().eq("").sum())
                                for col in ["Name", "Description"] if col in frame},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_pet_ids": int(frame.PetID.duplicated().sum()) if "PetID" in frame else None,
        "repeated_nonempty_descriptions": int(nonempty.duplicated().sum()),
        "nonempty_descriptions_with_non_ascii": int(nonempty.str.contains(r"[^\x00-\x7f]", regex=True).sum()),
        "invalid_code_counts": {
            col: int((frame[col].notna() & ~frame[col].isin(valid)).sum())
            for col, valid in CODE_RANGES.items() if col in frame
        },
        "negative_numeric_counts": {col: int((pd.to_numeric(frame[col], errors="coerce") < 0).sum())
                                    for col in NUMERIC if col in frame},
        "non_numeric_counts": {
            col: int((frame[col].notna() & pd.to_numeric(frame[col], errors="coerce").isna()).sum())
            for col in NUMERIC + CATEGORICAL if col in frame
        },
        "semantic_unknown_counts": {
            col: int(frame[col].eq(code).sum())
            for col, code in {"MaturitySize": 0, "FurLength": 0, "Health": 0,
                              "Vaccinated": 3, "Dewormed": 3, "Sterilized": 3}.items() if col in frame
        },
    }
    if "RescuerID" in frame:
        counts = frame.RescuerID.value_counts()
        result["rescuer_groups"] = {"n_unique": len(counts), "largest_group": int(counts.max()) if len(counts) else 0,
                                    "singleton_groups": int(counts.eq(1).sum()),
                                    "missing": int(frame.RescuerID.isna().sum())}
    if labeled and TARGET in frame:
        result["class_counts"] = {str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()}
        result["class_proportions"] = {
            str(k): float(v) for k, v in frame[TARGET].value_counts(normalize=True).sort_index().items()}
        result["invalid_target_count"] = int((~frame[TARGET].isin(CLASSES)).sum())
        result["descriptions_with_conflicting_labels"] = int(
            frame.assign(normalized_description=description).loc[description.ne("")]
            .groupby("normalized_description")[TARGET].nunique().gt(1).sum())
    return result


def modality_inventory(root, frame, split):
    """Обзор файла, json (но без эмбеддингв)"""
    root = Path(root)
    result = {}
    for modality, suffix in [("images", ".jpg"), ("metadata", ".json"), ("sentiment", ".json")]:
        candidates = [root / f"{split}_{modality}", root / split / modality]
        folder = next((p for p in candidates if p.is_dir()), None)
        if folder is None:
            result[modality] = {"available": False, "searched": [str(p) for p in candidates]}
            continue
        paths = sorted(folder.rglob(f"*{suffix}"))
        ids = [path.stem if modality == "sentiment" else path.stem.rsplit("-", 1)[0] for path in paths]
        counts = pd.Series(ids, dtype=str).value_counts()
        coverage = frame.PetID.astype(str).map(counts).fillna(0)
        known_ids = set(frame.PetID.astype(str))
        info = {"available": True, "folder": str(folder), "files": len(paths),
                "rows_with_files": int(coverage.gt(0).sum()), "rows_without_files": int(coverage.eq(0).sum()),
                "files_for_unknown_pet_ids": int(sum(value for key, value in counts.items()
                                                      if key not in known_ids))}
        if modality == "images" and "PhotoAmt" in frame:
            info["rows_with_photo_count_mismatch"] = int(coverage.ne(frame.PhotoAmt).sum())
        if suffix == ".json":
            failures, keys = [], set()
            for path in paths[:100]:
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(obj, dict):
                        keys.update(obj.keys())
                    else:
                        failures.append(path.name)
                except (OSError, ValueError):
                    failures.append(path.name)
            info.update(json_files_sampled=min(100, len(paths)),
                        sample_parse_failures=failures, sampled_top_level_keys=sorted(keys))
        result[modality] = info
    return result


def run_eda(train, test, output, root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"train": frame_audit(train, True)}
    schema = pd.DataFrame({"column": train.columns, "dtype": train.dtypes.astype(str).values,
                           "missing": train.isna().sum().values,
                           "unique": train.nunique(dropna=False).values})
    schema.to_csv(output / "schema.csv", index=False)
    available_numeric = [col for col in NUMERIC if col in train]
    train[available_numeric].describe(percentiles=[0.01, 0.5, 0.99]).to_csv(output / "numeric_summary.csv")
    if test is not None:
        report["test"] = frame_audit(test, False)
        report["train_test"] = {
            "overlapping_pet_ids": len(set(train.PetID) & set(test.PetID)),
            "test_rows_with_train_rescuer": int(test.RescuerID.isin(train.RescuerID).sum()),
            "test_rows_with_train_description": int(normalized_description(test).isin(
                set(normalized_description(train)) - {""}).sum()),
            "unseen_category_rows_in_test": {
                col: int((test[col].notna() & ~test[col].isin(train[col].dropna())).sum())
                for col in CATEGORICAL if col in test and col in train},
        }
    if root is not None:
        report["train_modalities"] = modality_inventory(root, train, "train")
        if test is not None:
            report["test_modalities"] = modality_inventory(root, test, "test")
    save_json(report, output / "audit.json")
    if TARGET in train:
        counts = train[TARGET].value_counts().reindex(CLASSES, fill_value=0)
        pd.DataFrame({"count": counts, "fraction": counts / len(train)}).rename_axis(TARGET).to_csv(
            output / "class_counts.csv")
        fig, ax = plt.subplots(figsize=(6, 3))
        counts.plot.bar(ax=ax, rot=0)
        ax.set(xlabel="AdoptionSpeed", ylabel="Listings", title="Target distribution (train)")
        fig.tight_layout()
        fig.savefig(output / "class_distribution.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(available_numeric), figsize=(15, 3), squeeze=False)
    for ax, col in zip(axes[0], available_numeric):
        values = pd.to_numeric(train[col], errors="coerce").dropna()
        ax.hist(np.log1p(values.clip(lower=0)), bins=30)
        ax.set_title(f"log1p({col})")
    fig.tight_layout()
    fig.savefig(output / "numeric_distributions.png", dpi=140)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="outputs/eda")
    args = parser.parse_args()

    train = pd.read_csv(find_csv(args.data_root, "train.csv"))
    test = pd.read_csv(find_csv(args.data_root, "test.csv"))
    report = run_eda(train, test, args.output, args.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
