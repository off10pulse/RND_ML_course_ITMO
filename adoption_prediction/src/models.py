"""Модельки и препроцессинг"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC
from .photo_features import PHOTO_FEATURES

MODEL_NAMES = ["dummy", "logreg_tabular", "logreg_text", "catboost_regressor",
               "catboost_classifier"]


def prepare_features(frame, photo_metadata=False):
    result = frame[NUMERIC + CATEGORICAL].copy()
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="raise").astype(float)
    for col in CATEGORICAL:
        result[col] = pd.to_numeric(result[col], errors="raise").map(
            lambda value: "__MISSING__" if pd.isna(value) else str(int(value))
        )
    result["Description"] = "petdescription " + frame.Description.fillna("").astype(str)
    if photo_metadata:
        for col in PHOTO_FEATURES:
            result[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    return result


def tabular_only(frame, photo_metadata=False):
    return frame[NUMERIC + CATEGORICAL + (PHOTO_FEATURES if photo_metadata else [])]


def make_model(name, config):
    seed = config.get("seed", 42)
    if name not in MODEL_NAMES:
        raise ValueError(f"Неизвестная модель {name}, нужно выбрать из: {MODEL_NAMES}")
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    use_photo = config.get("features", {}).get("photo_metadata", False)
    if use_photo and not name.startswith("catboost"):
        raise ValueError("Метаданные фото в этих экспериментах поддерживаются только CatBoost")
    prepare = FunctionTransformer(prepare_features, validate=False,
                                 kw_args={"photo_metadata": use_photo})
    if name.startswith("logreg"):
        numerical = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transforms = [
            ("numeric", numerical, NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
        if name == "logreg_text":
            text_config = config.get("text", {})
            transforms.append(("text", TfidfVectorizer(
                ngram_range=(1, 2), min_df=1,
                max_features=text_config.get("max_features", 20000),
                sublinear_tf=True, strip_accents="unicode",
            ), "Description"))
        params = config.get("logreg", {})
        return Pipeline([
            ("prepare", prepare),
            ("encode", ColumnTransformer(transforms, sparse_threshold=1.0)),
            ("model", LogisticRegression(
                C=params.get("C", 1.0), solver="lbfgs",
                max_iter=params.get("max_iter", 1500), random_state=seed,
            )),
        ])
    from catboost import CatBoostClassifier, CatBoostRegressor
    params = dict(config.get("catboost", {}))
    params.update(random_seed=seed, cat_features=CATEGORICAL,
                  verbose=False, allow_writing_files=False, task_type="CPU")
    if name == "catboost_regressor":
        estimator = CatBoostRegressor(loss_function="RMSE", **params)
    else:
        estimator = CatBoostClassifier(loss_function="MultiClass", **params)
    return Pipeline([
        ("prepare", prepare),
        ("select", FunctionTransformer(tabular_only, validate=False,
                                      kw_args={"photo_metadata": use_photo})),
        ("model", estimator),
    ])


def prediction_labels(model, frame, name):
    values = np.asarray(model.predict(frame)).reshape(-1)
    if name == "catboost_regressor":
        return np.digitize(values, [0.5, 1.5, 2.5, 3.5]).astype(int)
    return values.astype(int)
