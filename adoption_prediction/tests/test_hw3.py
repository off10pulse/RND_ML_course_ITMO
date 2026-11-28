from copy import deepcopy
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import diagnostics
from src.data import TARGET
from src.diagnostics import (class_metrics, error_examples, export_first_photos,
                             join_oof, noisy_age, run_age_noise, slice_metrics)
from src.models import make_model
from src.train import make_folds


def sample_oof(pets, config):
    folds, _ = make_folds(pets, config)
    prediction = pets.AdoptionSpeed.map({0: 1, 1: 4, 2: 2, 3: 2, 4: 2})
    return pets[["PetID", "RescuerID", "Type", "Age", "Quantity", "PhotoAmt"]].assign(
        fold=folds, y_true=pets.AdoptionSpeed, prediction=prediction)


def test_oof_alignment_and_changed_input(pets, config):
    oof = sample_oof(pets, config).sample(frac=1, random_state=2)
    joined = join_oof(pets, oof)
    assert joined.PetID.tolist() == pets.PetID.tolist()
    assert joined.prediction.tolist() == pets.AdoptionSpeed.map({0: 1, 1: 4, 2: 2, 3: 2, 4: 2}).tolist()
    changed = oof.copy()
    changed.loc[changed.index[0], "y_true"] = 99
    with pytest.raises(ValueError, match="target"):
        join_oof(pets, changed)
    changed = oof.copy()
    changed.loc[changed.index[0], "Age"] = 999
    with pytest.raises(ValueError, match="Age"):
        join_oof(pets, changed)
    with pytest.raises(ValueError, match="PetID"):
        join_oof(pets, oof.iloc[1:])


def test_slices_cover_data_and_examples_are_reproducible(pets, config):
    joined = join_oof(pets, sample_oof(pets, config))
    table = slice_metrics(joined)
    assert table.groupby("slice").n.sum().eq(len(pets)).all()
    assert np.allclose(table[[f"class_{c}_share" for c in range(5)]].sum(axis=1), 1)
    per_class = class_metrics(joined)
    assert per_class.loc[per_class["class"].eq(0), "recall"].iloc[0] == 0
    frequency, examples = error_examples(joined)
    assert len(examples) == 4
    assert frequency.n_errors.eq(12).all()
    shuffled = joined.sample(frac=1, random_state=3)
    _, other = error_examples(shuffled)
    assert examples.PetID.tolist() == other.PetID.tolist()
    perfect = joined.assign(prediction=joined.y_true, absolute_error=0)
    empty_frequency, empty_examples = error_examples(perfect)
    assert empty_examples.empty and "PetID" in empty_examples
    assert empty_frequency.n_errors.eq(0).all()


def test_split_seed_is_independent_of_model_seed(pets, config):
    # При одинаковом составе всех групп SGKF может выдать те же фолды с любым seed.
    frame = pets.loc[pets.index % 7 != 0].copy()
    first = deepcopy(config)
    first["split_seed"] = 17
    second = deepcopy(first)
    second["split_seed"] = 2026
    assert not np.array_equal(make_folds(frame, first)[0], make_folds(frame, second)[0])
    assert make_model("catboost_classifier", first).named_steps["model"].get_param("random_seed") == 42
    assert make_model("catboost_classifier", second).named_steps["model"].get_param("random_seed") == 42
    second["seed"] = 123
    assert np.array_equal(make_folds(frame, {**second, "seed": 42})[0], make_folds(frame, second)[0])


def test_first_photo_export_reads_only_selected_images(tmp_path):
    archive = tmp_path / "pets.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("wrapper/train_images/a-1.jpg", b"first")
        zipped.writestr("wrapper/train_images/a-2.jpg", b"second")
        zipped.writestr("wrapper/test_images/a-1.jpg", b"test")
    found = export_first_photos(archive, ["a", "missing"], tmp_path / "images")
    assert set(found) == {"a"}
    assert found["a"].read_bytes() == b"first"
    assert len(list((tmp_path / "images").iterdir())) == 1


def test_age_noise_keeps_training_clean_and_validation_groups_separate(pets, config, tmp_path, monkeypatch):
    original = pets.copy(deep=True)
    original_by_id = pets.set_index("PetID")
    fitted = []

    class RecordingModel:
        def __init__(self, name, settings):
            self.model = make_model(name, settings)

        def fit(self, frame, target):
            pd.testing.assert_frame_equal(frame.set_index("PetID"), original_by_id.loc[frame.PetID])
            self.rescuers = set(frame.RescuerID)
            fitted.append(frame.PetID.tolist())
            self.model.fit(frame, target)
            return self

        def predict(self, frame):
            assert not self.rescuers & set(frame.RescuerID)
            actual = frame.set_index("PetID").drop(columns="Age")
            pd.testing.assert_frame_equal(actual, original_by_id.loc[frame.PetID].drop(columns="Age"))
            return self.model.predict(frame)

    monkeypatch.setattr(diagnostics, "make_model", RecordingModel)
    folds, _ = make_folds(pets, config)
    saved = pets[["PetID", "RescuerID", TARGET]].assign(fold=folds)
    runs, summary = run_age_noise(pets, config, saved, tmp_path / "noise", sigmas=[0, 1, 3], seeds=[123, 124])
    assert len(fitted) == config["n_splits"]
    assert len(runs) == 5 and len(summary) == 3
    assert summary.loc[summary.sigma.eq(0), "delta_mean_qwk"].iloc[0] == 0
    assert summary.loc[summary.sigma.gt(0), "changed_age_rate"].gt(0).all()
    clean = pd.read_csv(tmp_path / "noise/oof_clean.csv")
    assert clean.Age.equals(clean.noisy_age)
    assert clean.fold.tolist() == folds.tolist()
    assert (tmp_path / "noise/oof_sigma_3_seed_124.csv").is_file()
    pd.testing.assert_frame_equal(pets, original)
    extreme = noisy_age(pets, 100, np.full(len(pets), -100))
    assert extreme.Age.eq(0).all()
    with pytest.raises(ValueError):
        run_age_noise(pets, config, saved, tmp_path / "bad", sigmas=[1], seeds=[123])
