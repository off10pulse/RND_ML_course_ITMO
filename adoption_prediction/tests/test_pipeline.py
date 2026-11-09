import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import load_data, validate_frame
from src.eda import run_eda
from src.evaluate import group_bootstrap, metrics
from src.models import make_model, prediction_labels, prepare_features
from src.report import build_report
from src.train import fit_submission, make_folds, run_cv


def test_group_split_and_determinism(pets, config):
    folds, diagnostics = make_folds(pets, config)
    assert np.array_equal(folds, make_folds(pets, config)[0])
    assert len(set(folds)) == 2
    assert all(item["overlapping_rescuers"] == 0 for item in diagnostics)
    assert pets.assign(fold=folds).groupby("RescuerID").fold.nunique().max() == 1


def test_train_only_preprocessing_and_no_ids(pets, config):
    train, val = pets.iloc[:40].copy(), pets.iloc[40:].copy()
    val["Description"] = "validationexclusiveword"
    val["State"] = 999999
    assert not {"PetID", "RescuerID", "AdoptionSpeed", "Name"} & set(prepare_features(train))
    model = make_model("logreg_text", config).fit(train, train.AdoptionSpeed)
    vocabulary = model.named_steps["encode"].named_transformers_["text"].vocabulary_
    assert "validationexclusiveword" not in vocabulary
    imputer = model.named_steps["encode"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == train.Age.median()
    assert len(prediction_labels(model, val, "logreg_text")) == len(val)


def test_fixed_ordinal_boundaries():
    class Regressor:
        def predict(self, frame):
            return np.array([-10, 0.5, 1.49, 2.5, 9])
    assert prediction_labels(Regressor(), None, "catboost_regressor").tolist() == [0, 1, 1, 3, 4]
    assert metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])["qwk"] == 1.0
    assert abs(metrics([0, 1, 2, 3, 4], [4] * 5)["qwk"]) < 1e-12
    with pytest.raises(ValueError):
        metrics([0, 1], [0.2, 1])


def test_input_layout_validation_and_eda(pets, tmp_path):
    root = tmp_path / "raw"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    test = pets.drop(columns="AdoptionSpeed").copy()
    test.PetID = "test-" + test.PetID
    pets.to_csv(root / "train/train.csv", index=False)
    test.to_csv(root / "test/test.csv", index=False)
    train, test = load_data(root)
    sentiment = root / "train_sentiment"
    sentiment.mkdir()
    (sentiment / f"{train.PetID.iloc[0]}.json").write_text('{"documentSentiment": {}}')
    (sentiment / f"{train.PetID.iloc[1]}.json").write_text("broken")
    report = run_eda(train, test, tmp_path / "eda", root)
    assert report["train"]["shape"] == [len(pets), len(pets.columns)]
    assert report["train_modalities"]["sentiment"]["rows_without_files"] == len(pets) - 2
    assert len(report["train_modalities"]["sentiment"]["sample_parse_failures"]) == 1
    assert (tmp_path / "eda/class_distribution.png").is_file()
    broken = pets.copy()
    broken.loc[0, "RescuerID"] = None
    with pytest.raises(ValueError, match="Missing RescuerID"):
        validate_frame(broken)


def test_all_models_cv_report_and_refit(pets, config, tmp_path):
    result = run_cv(pets, config, tmp_path / "run")
    assert set(result.model) == set(config["models"])
    assert result.qwk_mean.notna().all()
    for name in config["models"]:
        oof = pd.read_csv(tmp_path / "run" / name / "oof.csv")
        assert oof.PetID.tolist() == pets.PetID.tolist()
        assert oof.prediction.between(0, 4).all()
        assert len(oof) == len(pets)
    assert build_report(tmp_path / "run").is_file()
    test = pets.drop(columns="AdoptionSpeed").iloc[:5].copy()
    test.PetID = "test-" + test.PetID
    sample_path = tmp_path / "sample.csv"
    test[["PetID"]].iloc[::-1].to_csv(sample_path, index=False)
    for name in config["models"]:
        destination = tmp_path / "refit" / name
        submission = fit_submission(pets, test, config, name, destination, sample_path)
        assert submission.PetID.tolist() == test.PetID.iloc[::-1].tolist()
        assert submission.AdoptionSpeed.between(0, 4).all()
        restored = joblib.load(destination / "model.joblib")
        assert len(prediction_labels(restored, test, name)) == len(test)


def test_paired_group_bootstrap(pets):
    predictions = pets[["RescuerID"]].copy()
    predictions["y_true"] = pets.AdoptionSpeed
    predictions["prediction"] = pets.AdoptionSpeed
    predictions["reference"] = pets.AdoptionSpeed
    interval = group_bootstrap(predictions, repeats=20, other="reference")
    assert interval["low"] == interval["high"] == 0
