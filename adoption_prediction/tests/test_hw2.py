from copy import deepcopy
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL, NUMERIC, TARGET
from src.models import make_model, prediction_labels
from src.photo_features import PHOTO_FEATURES, features_from_metadata, load_photo_features
from src.train import make_folds, run_cv


def vision_json():
    return {
        "labelAnnotations": [{"description": "Cat", "score": 0.9}],
        "imagePropertiesAnnotation": {"dominantColors": {"colors": [
            {"color": {"red": 100, "blue": 20}, "pixelFraction": 0.2},
            {"color": {"red": 200, "green": 40}, "pixelFraction": 0.6},
        ]}},
    }


def test_first_photo_zip_directory_and_missing_fields(tmp_path):
    payloads = {
        "train_metadata/pet-a-1.json": vision_json(),
        "train_metadata/pet-a-2.json": {"labelAnnotations": [{"description": "dog", "score": 1}]},
        "train_metadata/pet-b-1.json": {},
        "test_metadata/pet-a-1.json": {},
    }
    archive = tmp_path / "pets.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name, payload in payloads.items():
            path = tmp_path / "raw" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            zip_file.writestr("wrapper/" + name, json.dumps(payload))
    ids = ["pet-missing", "pet-a", "pet-b"]
    zipped = load_photo_features(archive, ids)
    unpacked = load_photo_features(tmp_path / "raw", ids)
    pd.testing.assert_frame_equal(zipped, unpacked)
    assert zipped.PetID.tolist() == ids
    missing, cat, empty = zipped.set_index("PetID").loc[ids].to_dict("records")
    assert missing["photo_metadata_missing"] == 1
    assert np.isnan(missing["photo_cat_score"])
    assert empty["photo_metadata_missing"] == 0 and empty["photo_labels_missing"] == 1
    assert cat["photo_cat_score"] == 0.9 and cat["photo_dog_score"] == 0
    assert cat["photo_red"] == pytest.approx(175)
    assert cat["photo_green"] == pytest.approx(30)
    assert cat["photo_blue"] == pytest.approx(5)
    assert load_photo_features(archive, ["pet-a"], split="test").photo_labels_missing.iloc[0] == 1
    assert load_photo_features(archive, ["no-photo"]).photo_metadata_missing.iloc[0] == 1


def test_photo_source_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_metadata"):
        load_photo_features(tmp_path, ["a"])
    directory = tmp_path / "train_metadata"
    directory.mkdir()
    (directory / "a-1.json").write_text("broken JSON")
    with pytest.raises(ValueError, match="a-1.json"):
        load_photo_features(tmp_path, ["a"])
    with pytest.raises(ValueError, match="PetID"):
        load_photo_features(tmp_path, [None])
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("one/train_metadata/a-1.json", "{}")
        zip_file.writestr("two/train_metadata/a-1.json", "{}")
    with pytest.raises(ValueError, match="Повторный JSON"):
        load_photo_features(archive, ["a"])


def test_saved_folds_align_by_id_and_reject_changed_labels_or_groups(pets, config):
    original, _ = make_folds(pets, config)
    saved = pets[["PetID", "RescuerID", TARGET]].assign(fold=original)
    shuffled = pets.sample(frac=1, random_state=1)
    restored, _ = make_folds(shuffled, config, folds=saved.iloc[::-1])
    expected = saved.set_index("PetID").loc[shuffled.PetID, "fold"].to_numpy()
    assert np.array_equal(restored, expected)
    changed = saved.copy()
    changed.loc[0, TARGET] = (changed.loc[0, TARGET] + 1) % 5
    with pytest.raises(ValueError, match=TARGET):
        make_folds(pets, config, folds=changed)
    changed = saved.copy()
    changed.loc[0, "fold"] = 1 - changed.loc[0, "fold"]
    with pytest.raises(AssertionError):
        make_folds(pets, config, folds=changed)
    with pytest.raises(ValueError, match="PetID"):
        make_folds(pets.iloc[1:], config, folds=saved)


def test_photo_ablation_and_fold_local_weights(pets, config, tmp_path):
    frame = pets.copy()
    photo = features_from_metadata(vision_json())
    for column in PHOTO_FEATURES:
        frame[column] = photo[column]
    frame.loc[0, PHOTO_FEATURES] = list(features_from_metadata(None).values())
    base_config = deepcopy(config)
    base_config["models"] = ["catboost_classifier"]
    base_config["bootstrap_repeats"] = 0
    enriched = deepcopy(base_config)
    enriched["features"] = {"photo_metadata": True}
    enriched["catboost"]["auto_class_weights"] = "SqrtBalanced"

    # Обучающая часть специально несбалансирована, отложенные строки не участвуют в весах.
    training = frame.iloc[:40].copy()
    training = training[~((training.AdoptionSpeed == 0) & (training.index > 5))]
    model = make_model("catboost_classifier", enriched).fit(training, training.AdoptionSpeed)
    estimator = model.named_steps["model"]
    counts = training.AdoptionSpeed.value_counts().sort_index()
    assert np.allclose(estimator.get_all_params()["class_weights"], np.sqrt(counts.max() / counts))
    assert set(estimator.feature_names_) == set(NUMERIC + CATEGORICAL + PHOTO_FEATURES)
    assert not {"PetID", "RescuerID", TARGET, "Name", "Description"} & set(estimator.feature_names_)
    validation = frame.iloc[40:]
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert np.array_equal(prediction_labels(model, validation, "catboost_classifier"),
                          prediction_labels(joblib.load(path), validation, "catboost_classifier"))
    baseline = make_model("catboost_classifier", base_config).fit(training, training.AdoptionSpeed)
    assert set(baseline.named_steps["model"].feature_names_) == set(NUMERIC + CATEGORICAL)
    original = make_model("catboost_classifier", base_config).fit(pets, pets.AdoptionSpeed)
    no_photo = make_model("catboost_classifier", base_config).fit(frame, frame.AdoptionSpeed)
    assert np.array_equal(original.predict(pets), no_photo.predict(frame))

    folds, _ = make_folds(frame, base_config)
    saved = frame[["PetID", "RescuerID", TARGET]].assign(fold=folds)
    run_cv(frame, enriched, tmp_path / "cv", folds=saved)
    actual = pd.read_csv(tmp_path / "cv/folds.csv")
    assert actual.fold.tolist() == saved.fold.tolist()
