"""перерасчет метрик без обучения"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES
from src.evaluate import metrics, save_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_run(run_dir, refit_dir=None, expected_test_rows=None):
    run_dir = Path(run_dir)
    summary = pd.read_csv(run_dir / "summary.csv")
    scores = pd.read_csv(run_dir / "fold_metrics.csv")
    folds = pd.read_csv(run_dir / "folds.csv")
    require(folds.PetID.is_unique, "Duplicate PetID in folds")
    require(folds.groupby("RescuerID").fold.nunique().max() == 1, "Rescuer crosses folds")
    require(summary.model.is_unique, "Duplicate model in summary")
    require(set(scores.model) == set(summary.model), "Model lists differ")
    result = {"rows": len(folds), "rescuer_groups": folds.RescuerID.nunique(),
              "folds": folds.fold.nunique(), "models": {},
              "class_counts": {str(k): int(v) for k, v in folds.AdoptionSpeed.value_counts().sort_index().items()}}
    for row in summary.itertuples(index=False):
        name = row.model
        oof = pd.read_csv(run_dir / name / "oof.csv")
        require(oof.PetID.is_unique and set(oof.PetID) == set(folds.PetID), f"{name}: OOF coverage")
        joined = oof.merge(folds, on="PetID", suffixes=("", "_expected"), validate="one_to_one")
        for actual, expected in [("fold", "fold_expected"), ("RescuerID", "RescuerID_expected"),
                                 ("y_true", "AdoptionSpeed")]:
            require(joined[actual].eq(joined[expected]).all(), f"{name}: {actual} mismatch")
        require(oof.absolute_error.eq((oof.y_true - oof.prediction).abs()).all(), f"{name}: error mismatch")
        measured = metrics(oof.y_true, oof.prediction)
        for key, value in measured.items():
            require(np.isclose(value, getattr(row, "oof_" + key), rtol=0, atol=1e-12), f"{name}: OOF {key}")
        per_fold = scores[scores.model.eq(name)]
        require(set(per_fold.fold) == set(folds.fold) and per_fold.fold.is_unique, f"{name}: fold coverage")
        for score in per_fold.itertuples(index=False):
            subset = oof[oof.fold.eq(score.fold)]
            for key, value in metrics(subset.y_true, subset.prediction).items():
                require(np.isclose(value, getattr(score, key), rtol=0, atol=1e-12), f"{name}: fold {score.fold} {key}")
        require(np.isclose(per_fold.qwk.mean(), row.qwk_mean, rtol=0, atol=1e-12), f"{name}: QWK mean")
        require(np.isclose(per_fold.qwk.std(ddof=1), row.qwk_std, rtol=0, atol=1e-12), f"{name}: QWK std")
        saved_matrix = pd.read_csv(run_dir / name / "confusion_matrix.csv", index_col=0).values
        require(np.array_equal(saved_matrix, confusion_matrix(oof.y_true, oof.prediction, labels=CLASSES)),
                f"{name}: confusion matrix")
        errors = pd.read_csv(run_dir / name / "errors.csv")
        require(errors.PetID.is_unique and errors.PetID.isin(oof.PetID).all(), f"{name}: error IDs")
        expected = oof.set_index("PetID").loc[errors.PetID].reset_index()
        require(expected[errors.columns].equals(errors), f"{name}: error examples differ from OOF")
        if len(errors):
            cutoff = oof.absolute_error.nlargest(len(errors)).min()
            require(errors.absolute_error.min() >= cutoff, f"{name}: errors are not the largest ones")
        slices_path = run_dir / name / "slices.csv"
        if slices_path.stat().st_size > 1:
            for item in pd.read_csv(slices_path).itertuples(index=False):
                subset = oof[oof[item.column].eq(item.value)]
                require(len(subset) == item.n, f"{name}: slice size {item.column}={item.value}")
                for key, value in metrics(subset.y_true, subset.prediction).items():
                    saved = getattr(item, key)
                    agrees = pd.isna(saved) if value is None else np.isclose(value, saved, rtol=0, atol=1e-12)
                    require(agrees, f"{name}: slice {item.column}={item.value} {key}")
        result["models"][name] = {"metrics_recomputed": True, "oof_rows": len(oof),
                                  "prediction_counts": {str(k): int(v) for k, v in oof.prediction.value_counts().sort_index().items()}}
    if refit_dir is not None:
        refit_dir = Path(refit_dir)
        refit = json.loads((refit_dir / "refit.json").read_text())
        require(refit["model"] == summary.sort_values("qwk_mean", ascending=False).iloc[0].model,
                "Refit model differs from CV winner")
        submission = pd.read_csv(refit_dir / "submission.csv")
        require(list(submission.columns) == ["PetID", "AdoptionSpeed"], "Submission columns")
        require(submission.PetID.notna().all() and submission.PetID.is_unique, "Submission IDs")
        require(submission.AdoptionSpeed.isin(CLASSES).all(), "Submission labels")
        if expected_test_rows is not None:
            require(len(submission) == expected_test_rows, "Submission length")
        require(not set(submission.PetID) & set(folds.PetID), "Train/submission ID overlap")
        result["refit"] = {"model": refit["model"], "submission_rows": len(submission),
                           "unique_ids_and_valid_labels": True,
                           "test_id_alignment_independently_verified": False,
                           "note": "Original test/sample_submission not available for independent ID/order check; model was not deserialized."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--refit-dir")
    parser.add_argument("--expected-test-rows", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_run(args.run_dir, args.refit_dir, args.expected_test_rows)
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
