"""Метрики и бутстрэп"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error, recall_score)

from .data import CLASSES


def metrics(y_true, y_pred):
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if len(truth) != len(pred) or not len(truth):
        raise ValueError("Пустые предикты или нет связи с таргетом")
    if not np.isin(truth, CLASSES).all() or not np.isin(pred, CLASSES).all():
        raise ValueError("Метки должны быть 0-4")
    value = float(cohen_kappa_score(truth, pred, labels=CLASSES, weights="quadratic"))
    return {
        "qwk": value if np.isfinite(value) else None,
        "macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, pred)),
        "mae": float(mean_absolute_error(truth, pred)),
        "recall_class_0": float(recall_score(truth, pred, labels=[0], average=None,
                                            zero_division=0)[0]),
        "severe_error_rate": float(np.mean(np.abs(truth - pred) >= 2)),
    }


def group_bootstrap(frame, repeats=500, seed=42, other=None):
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("RescuerID", sort=False).indices.values())
    values = []
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = frame.iloc[indices]
        score = metrics(sample.y_true, sample.prediction)["qwk"]
        if other is not None:
            reference = metrics(sample.y_true, sample[other])["qwk"]
            score = None if score is None or reference is None else score - reference
        if score is not None:
            values.append(score)
    return {"low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_repeats": len(values), "requested_repeats": repeats}


def save_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="One model's oof.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    result = {"metrics": metrics(frame.y_true, frame.prediction),
              "confusion_matrix": confusion_matrix(frame.y_true, frame.prediction,
                                                    labels=CLASSES).tolist()}
    print(json.dumps(result, indent=2))
    if args.output:
        save_json(result, args.output)


if __name__ == "__main__":
    main()
