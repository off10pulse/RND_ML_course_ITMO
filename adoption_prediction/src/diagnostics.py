"""Срезы, примеры ошибок и проверка чувствительности к возрасту для ДЗ3."""
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support

from .data import CLASSES, TARGET, file_sha256
from .evaluate import metrics, save_json
from .models import make_model, prediction_labels
from .train import environment, make_folds


def join_oof(frame, oof):
    """Присоединить OOF по PetID, проверив разметку и исходные признаки."""
    if not oof.PetID.is_unique or set(oof.PetID) != set(frame.PetID):
        raise ValueError("OOF не покрывает текущий train по PetID")
    aligned = oof.set_index("PetID").loc[frame.PetID]
    for column in ["RescuerID", "Type", "Age", "Quantity", "PhotoAmt"]:
        actual = pd.Series(aligned[column].to_numpy())
        expected = pd.Series(frame[column].to_numpy())
        if not (actual.eq(expected) | (actual.isna() & expected.isna())).all():
            raise ValueError(f"OOF: отличается {column}")
    if not np.array_equal(aligned.y_true, frame[TARGET]):
        raise ValueError("OOF: отличается target")
    if not aligned.prediction.isin(CLASSES).all() or aligned.fold.isna().any():
        raise ValueError("Некорректные предсказания или фолды OOF")
    result = frame.copy()
    for column in ["fold", "y_true", "prediction"]:
        result[column] = aligned[column].to_numpy()
    result["absolute_error"] = (result.y_true - result.prediction).abs()
    return result


def slice_metrics(frame):
    dimensions = {
        "species": frame.Type.map({1: "Собаки", 2: "Кошки"}).fillna("Неизвестно"),
        "age": pd.cut(frame.Age, [-np.inf, 2, 11, np.inf],
                      labels=["0–2 месяца", "3–11 месяцев", "12+ месяцев"]),
        "quantity": pd.Series(np.where(frame.Quantity > 1, "Несколько животных", "Одно животное"), index=frame.index),
        "photos": pd.Series(np.where(frame.PhotoAmt > 0, "Есть фото", "Нет фото"), index=frame.index),
    }
    rows = []
    for dimension, groups in dimensions.items():
        groups = groups.astype(object).where(groups.notna(), "Неизвестно")
        order = (["0–2 месяца", "3–11 месяцев", "12+ месяцев", "Неизвестно"]
                 if dimension == "age" else sorted(groups.unique()))
        for group in order:
            subset = frame.loc[groups.eq(group)]
            if subset.empty:
                continue
            counts = subset.y_true.value_counts()
            with np.errstate(divide="ignore", invalid="ignore"):
                score = metrics(subset.y_true, subset.prediction)
            if subset.y_true.nunique() < 2:
                score["qwk"] = None
            rows.append({"slice": dimension, "group": group, "n": len(subset),
                         "rescuer_groups": int(subset.RescuerID.nunique()), **score,
                         **{f"class_{c}_share": float(counts.get(c, 0) / len(subset)) for c in CLASSES}})
    return pd.DataFrame(rows)


def class_metrics(frame):
    precision, recall, f1, support = precision_recall_fscore_support(
        frame.y_true, frame.prediction, labels=CLASSES, zero_division=0)
    return pd.DataFrame({"class": CLASSES, "n": support,
                         "precision": precision, "recall": recall, "f1": f1})


def error_examples(frame):
    definitions = [
        ("same_day", "Не распознан класс 0", frame.y_true.eq(0), frame.prediction.ne(0)),
        ("adjacent", "Соседний интервал для класса 3", frame.y_true.eq(3), frame.absolute_error.eq(1)),
        ("optimistic", "Слишком оптимистичный прогноз", frame.y_true.eq(4), frame.prediction.le(2)),
        ("pessimistic", "Слишком пессимистичный прогноз", frame.y_true.eq(1), frame.prediction.ge(3)),
    ]
    frequency, examples = [], []
    for key, label, target, error in definitions:
        subset = frame.loc[target & error]
        frequency.append({"pattern": key, "description": label, "n_errors": len(subset),
                          "n_target_class": int(target.sum()),
                          "share_in_target_class": float(len(subset) / target.sum()) if target.any() else None})
        if subset.empty:
            continue
        # Самая частая пара y -> prediction, затем ближайший к медианному возраст.
        pair = subset.groupby(["y_true", "prediction"]).size().sort_values(ascending=False, kind="stable").index[0]
        candidates = subset.loc[subset.y_true.eq(pair[0]) & subset.prediction.eq(pair[1])].copy()
        candidates["age_distance"] = (candidates.Age - candidates.Age.median()).abs()
        chosen = candidates.sort_values(["age_distance", "PetID"], kind="stable").iloc[0].drop("age_distance")
        examples.append({"pattern": key, "pattern_description": label, **chosen.to_dict()})
    return pd.DataFrame(frequency), pd.DataFrame(examples, columns=["pattern", "pattern_description", *frame.columns])


def export_first_photos(source, pet_ids, output):
    """Скопировать только выбранные JPG первого фото из ZIP или полной папки."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = [str(pet_id) for pet_id in pet_ids]
    if any(PurePosixPath(pet_id).name != pet_id or "\\" in pet_id for pet_id in ids):
        raise ValueError("Небезопасный PetID для имени фото")
    wanted = {f"{pet_id}-1.jpg": pet_id for pet_id in ids}
    found = {}
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if not member.is_dir() and path.parent.name == "train_images" and path.name in wanted:
                    pet_id = wanted[path.name]
                    if pet_id in found:
                        raise ValueError(f"Повторное первое фото: {pet_id}")
                    target = output / path.name
                    target.write_bytes(archive.read(member))
                    found[pet_id] = target
    else:
        for name, pet_id in wanted.items():
            path = source / "train_images" / name
            if path.is_file():
                target = output / name
                target.write_bytes(path.read_bytes())
                found[pet_id] = target
    return found


def noisy_age(frame, sigma, noise):
    if sigma < 0 or len(noise) != len(frame):
        raise ValueError("Некорректный уровень или длина шума")
    result = frame.copy()
    if sigma:
        result["Age"] = np.maximum(0, np.rint(frame.Age.to_numpy(dtype=float) + sigma * np.asarray(noise)))
    return result


def run_age_noise(frame, config, folds, output, sigmas=(0, 1, 3), seeds=(123, 124, 125)):
    """Пять обучений на чистом train, разные условия только при predict validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sigmas, seeds = list(sigmas), list(seeds)
    if (not sigmas or sigmas[0] != 0 or len(set(sigmas)) != len(sigmas)
            or any(not np.isfinite(sigma) or sigma < 0 for sigma in sigmas)
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("Нужны уникальные sigma >= 0 с первым 0 и непустые уникальные seeds")
    fold_ids, diagnostics = make_folds(frame, config, folds=folds)
    fold_table = frame[["PetID", "RescuerID", TARGET]].assign(fold=fold_ids)
    fold_table.to_csv(output / "folds.csv", index=False)
    save_json(diagnostics, output / "split_diagnostics.json")
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    save_json({"sigmas": sigmas, "noise_seeds": seeds, "train_age_changed": False,
               "rule": "max(0, round(Age + sigma * N(0, 1))) only on validation"}, output / "conditions.json")
    save_json({"environment": environment(), "source_sha256": {
        p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}, output / "environment.json")
    conditions = [(0, None)] + [(sigma, seed) for sigma in sigmas if sigma for seed in seeds]
    noise_vectors = {seed: np.random.default_rng(seed).standard_normal(len(frame)) for seed in seeds}
    predictions = {condition: np.full(len(frame), -1, dtype=int) for condition in conditions}
    fold_scores, durations = [], []
    save_json({"status": "running"}, output / "status.json")
    try:
        for fold in sorted(set(fold_ids)):
            train_idx, val_idx = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            model = make_model("catboost_classifier", config)
            started = time.perf_counter()
            model.fit(frame.iloc[train_idx], frame.iloc[train_idx][TARGET])
            durations.append(time.perf_counter() - started)
            validation = frame.iloc[val_idx]
            for sigma, seed in conditions:
                noise = np.zeros(len(val_idx)) if seed is None else noise_vectors[seed][val_idx]
                perturbed = noisy_age(validation, sigma, noise)
                pred = prediction_labels(model, perturbed, "catboost_classifier")
                predictions[(sigma, seed)][val_idx] = pred
                fold_scores.append({"sigma": sigma, "noise_seed": seed, "fold": int(fold),
                                    **metrics(validation[TARGET], pred)})
            pd.DataFrame(fold_scores).to_csv(output / "fold_metrics.csv", index=False)
            print(f"Age noise: fold={fold}, обучение {durations[-1]:.1f} с, все уровни шума проверены", flush=True)
        rows = []
        base = predictions[(0, None)]
        for sigma, seed in conditions:
            noise = np.zeros(len(frame)) if seed is None else noise_vectors[seed]
            age = noisy_age(frame, sigma, noise).Age.to_numpy()
            pred = predictions[(sigma, seed)]
            oof = frame[["PetID", "RescuerID", "Age"]].copy()
            oof = oof.assign(noisy_age=age, fold=fold_ids, y_true=frame[TARGET].to_numpy(),
                             prediction=pred, absolute_error=np.abs(frame[TARGET].to_numpy() - pred))
            filename = "oof_clean.csv" if seed is None else f"oof_sigma_{sigma}_seed_{seed}.csv"
            oof.to_csv(output / filename, index=False)
            scores = pd.DataFrame([s for s in fold_scores if s["sigma"] == sigma and s["noise_seed"] == seed])
            rows.append({"sigma": sigma, "noise_seed": seed,
                         "qwk_mean": float(scores.qwk.mean()), "qwk_fold_std": float(scores.qwk.std(ddof=1)),
                         **{f"oof_{key}": value for key, value in metrics(frame[TARGET], pred).items()},
                         "changed_age_rate": float(np.mean(age != frame.Age.to_numpy())),
                         "prediction_flip_rate": float(np.mean(pred != base))})
        runs = pd.DataFrame(rows)
        runs.to_csv(output / "runs.csv", index=False)
        clean = runs.loc[runs.sigma.eq(0)].iloc[0]
        summaries = []
        for sigma, subset in runs.groupby("sigma", sort=False):
            summaries.append({"sigma": sigma, "repeats": len(subset),
                              "qwk_mean": float(subset.qwk_mean.mean()),
                              "qwk_std_between_noise_repeats": float(subset.qwk_mean.std(ddof=1)) if len(subset) > 1 else 0.0,
                              "delta_mean_qwk": float(subset.qwk_mean.mean() - clean.qwk_mean),
                              **{c: float(subset[c].mean()) for c in ["oof_qwk", "oof_mae", "oof_recall_class_0",
                                                                     "oof_severe_error_rate", "changed_age_rate", "prediction_flip_rate"]}})
        summary = pd.DataFrame(summaries)
        summary.to_csv(output / "summary.csv", index=False)
        save_json({"status": "complete", "trained_models": len(durations),
                   "fit_seconds": sum(durations)}, output / "status.json")
        return runs, summary
    except Exception as error:
        save_json({"status": "failed", "error": str(error)}, output / "status.json")
        raise
