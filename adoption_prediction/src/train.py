"""Обучение, кросс-валидация и результаты (артефакты)"""
import argparse
import hashlib
import importlib.metadata
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import (CLASSES, TARGET, file_sha256, find_csv, load_data,
                   normalized_description, validate_frame)
from .evaluate import group_bootstrap, metrics, save_json
from .models import MODEL_NAMES, make_model, prediction_labels


def environment():
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "catboost", "matplotlib",
                "PyYAML", "joblib"]
    return {"python": platform.python_version(), "packages": {
        package: importlib.metadata.version(package) for package in packages
    }}


def make_folds(frame, config):
    validate_frame(frame)
    count = int(config.get("n_splits", 5))
    seed = int(config.get("seed", 42))
    if count < 2 or frame[TARGET].value_counts().min() < count:
        raise ValueError("Для каждого класса нужно минимум n_splits строк")
    kind = config.get("split", "stratified_group")
    if kind == "stratified_group":
        if frame.RescuerID.nunique() < count:
            raise ValueError("Недостаточно RescuerID групп")
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    elif kind == "stratified":
        splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"чзх: {kind}")
    fold_ids = np.full(len(frame), -1, dtype=int)
    diagnostics = []
    descriptions = normalized_description(frame)
    iterator = (splitter.split(frame, frame[TARGET], groups=frame.RescuerID)
                if kind == "stratified_group" else splitter.split(frame, frame[TARGET]))
    for fold, (train_idx, val_idx) in enumerate(iterator):
        train, val = frame.iloc[train_idx], frame.iloc[val_idx]
        overlap = set(train.RescuerID) & set(val.RescuerID)
        if kind == "stratified_group" and overlap:
            raise AssertionError("Утечка?!?")
        if set(train[TARGET]) != set(CLASSES) or set(val[TARGET]) != set(CLASSES):
            raise ValueError(f"Фолд {fold} без таргета??; надо уменьшить n_splits")
        fold_ids[val_idx] = fold
        seen_text = set(descriptions.iloc[train_idx]) - {""}
        diagnostics.append({
            "fold": fold, "train_rows": len(train), "val_rows": len(val),
            "overlapping_rescuers": len(overlap),
            "val_rows_with_train_description": int(descriptions.iloc[val_idx].isin(seen_text).sum()),
            "train_class_counts": train[TARGET].value_counts().sort_index().to_dict(),
            "val_class_counts": val[TARGET].value_counts().sort_index().to_dict(),
        })
    if (fold_ids < 0).any():
        raise AssertionError("Неполный OOF")
    return fold_ids, diagnostics


def run_cv(frame, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = config.get("models", MODEL_NAMES[:-1])
    if not names or len(set(names)) != len(names) or set(names) - set(MODEL_NAMES):
        raise ValueError("Нужны модельки")
    fold_ids, split_report = make_folds(frame, config)
    fold_table = frame[["PetID", "RescuerID", TARGET]].copy()
    fold_table["fold"] = fold_ids
    fold_table.to_csv(output / "folds.csv", index=False)
    save_json(split_report, output / "split_diagnostics.json")
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    metadata = environment()
    metadata["frame_sha256"] = hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata["source_sha256"] = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    save_json(metadata, output / "environment.json")
    all_scores, summary = [], []
    for name in names:
        destination = output / name
        destination.mkdir(exist_ok=True)
        oof = np.full(len(frame), -1, dtype=int)
        model_scores = []
        for fold in sorted(set(fold_ids)):
            train_idx, val_idx = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            model = make_model(name, config)
            started = time.perf_counter()

            model.fit(frame.iloc[train_idx], frame.iloc[train_idx][TARGET])
            oof[val_idx] = prediction_labels(model, frame.iloc[val_idx], name)
            score = {"model": name, "fold": int(fold),
                     **metrics(frame.iloc[val_idx][TARGET], oof[val_idx]),
                     "seconds": time.perf_counter() - started}
            model_scores.append(score)
            all_scores.append(score)
            print(f"{name} fold={fold} QWK={score['qwk']}", flush=True)
        predictions = frame[["PetID", "RescuerID", "Type", "Age", "Quantity", "PhotoAmt"]].copy()
        predictions["fold"] = fold_ids
        predictions["y_true"] = frame[TARGET].to_numpy()
        predictions["prediction"] = oof
        predictions["absolute_error"] = np.abs(frame[TARGET].to_numpy() - oof)
        predictions.to_csv(destination / "oof.csv", index=False)

        predictions.sort_values("absolute_error", ascending=False).head(30).to_csv(
            destination / "errors.csv", index=False)
        pd.DataFrame(confusion_matrix(frame[TARGET], oof, labels=CLASSES),
                     index=CLASSES, columns=CLASSES).to_csv(destination / "confusion_matrix.csv")
        slices = []
        for col in ["Type", "Quantity", "PhotoAmt", "y_true"]:
            for value, subset in predictions.groupby(col):
                if len(subset) >= 20:
                    slices.append({"column": col, "value": value, "n": len(subset),
                                   **metrics(subset.y_true, subset.prediction)})
        pd.DataFrame(slices).to_csv(destination / "slices.csv", index=False)
        qwk_values = [row["qwk"] for row in model_scores if row["qwk"] is not None]
        row = {"model": name, "qwk_mean": float(np.mean(qwk_values)) if qwk_values else None,
               "qwk_std": float(np.std(qwk_values, ddof=1)) if len(qwk_values) > 1 else None,
               **{f"oof_{key}": val for key, val in metrics(frame[TARGET], oof).items()},
               "seconds": sum(item["seconds"] for item in model_scores)}
        repeats = config.get("bootstrap_repeats", 500)
        if repeats:
            save_json(group_bootstrap(predictions, repeats, config.get("seed", 42)),
                      destination / "qwk_bootstrap.json")
        summary.append(row)
    pd.DataFrame(all_scores).to_csv(output / "fold_metrics.csv", index=False)
    table = pd.DataFrame(summary).sort_values("qwk_mean", ascending=False)
    table.to_csv(output / "summary.csv", index=False)
    return table


def fit_submission(train, test, config, name, output, sample_path=None):
    validate_frame(train)
    validate_frame(test, train=False)
    model = make_model(name, config)
    model.fit(train, train[TARGET])
    submission = pd.DataFrame({"PetID": test.PetID.to_numpy(),
                               TARGET: prediction_labels(model, test, name)})
    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        if sample.PetID.duplicated().any() or set(sample.PetID) != set(test.PetID):
            raise ValueError("IDs отличаются от теста")
        submission = sample[["PetID"]].merge(submission, on="PetID", validate="one_to_one", sort=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    save_json({"model": name, "config": config, "environment": environment()},
              output / "refit.json")
    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="outputs/baseline")
    parser.add_argument("--refit-model", choices=MODEL_NAMES, default=None,
                        help="Explicit model, selected using local validation")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train, test = load_data(args.data_root, with_test=args.refit_model is not None)
    if args.refit_model:
        candidates = [Path(args.data_root) / "sample_submission.csv",
                      Path(args.data_root) / "test" / "sample_submission.csv"]
        sample = next((p for p in candidates if p.is_file()), None)
        fit_submission(train, test, config, args.refit_model, args.output, sample)
    else:
        table = run_cv(train, config, args.output)
        save_json({"train_path": str(find_csv(args.data_root, "train.csv")),
                   "train_sha256": file_sha256(find_csv(args.data_root, "train.csv"))},
                  Path(args.output) / "input.json")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
