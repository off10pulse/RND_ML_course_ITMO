"""Синтетика для тестаа."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CATEGORICAL


def synthetic_data(n_groups=12):
    rows = []
    for group in range(n_groups):
        for target in range(5):
            row = {col: 1 for col in CATEGORICAL}
            row.update(PetID=f"pet-{group}-{target}", RescuerID=f"rescuer-{group}",
                       Name="", Description=f"friendly animal agegroup{target} sample{group}",
                       Age=1 + target * 12, Quantity=1, Fee=target * 10,
                       VideoAmt=np.nan, PhotoAmt=group % 4, AdoptionSpeed=target)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def pets():
    return synthetic_data()


@pytest.fixture
def config():
    return {"seed": 42, "n_splits": 2, "split": "stratified_group",
            "models": ["dummy", "logreg_tabular", "logreg_text", "catboost_regressor", "catboost_classifier"],
            "bootstrap_repeats": 10, "logreg": {"C": 1.0, "max_iter": 1000},
            "text": {"max_features": 100},
            "catboost": {"iterations": 8, "depth": 3, "learning_rate": 0.1, "thread_count": 1}}
