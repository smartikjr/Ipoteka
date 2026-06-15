"""
Инференс модели кредитного скоринга и подготовка данных.

Модель обучена в src/train.py (LightGBM). Здесь:
- загрузка модели и метаданных;
- подготовка строк с корректными типами категорий;
- расчёт вероятности дефолта (PD) и решения по заявке;
- доступ к отложенной (test) выборке для страниц fairness/мониторинга.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .train import SCORING_CAT_COLS, load_borrowers_split


@lru_cache(maxsize=1)
def load_model():
    model = joblib.load(C.SCORING_MODEL_PATH)
    meta = json.loads(C.SCORING_META_PATH.read_text(encoding="utf-8"))
    return model, meta


def prepare_frame(rows: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Привести DataFrame к признакам модели с корректными категориями."""
    X = rows[meta["features"]].copy()
    for col in meta["categorical"]:
        X[col] = pd.Categorical(X[col], categories=meta["cat_levels"][col])
    return X


def predict_proba(rows: pd.DataFrame) -> np.ndarray:
    model, meta = load_model()
    X = prepare_frame(rows, meta)
    return model.predict_proba(X)[:, 1]


def predict_one(values: dict) -> dict:
    """Решение по одной заявке. Возвращает PD, балл и вердикт."""
    model, meta = load_model()
    row = pd.DataFrame([values])
    pd_value = float(predict_proba(row)[0])
    threshold = meta["threshold"]
    approved = pd_value < threshold
    # Балл 0..1000 (выше — надёжнее), монотонно убывает с ростом PD
    score = int(round(1000 * (1 - pd_value)))
    return {
        "pd": pd_value,
        "score": score,
        "approved": bool(approved),
        "threshold": threshold,
        "verdict": "Одобрено" if approved else "Отказано",
    }


@lru_cache(maxsize=1)
def get_scored_test() -> pd.DataFrame:
    """Отложенная выборка с рассчитанными PD и решениями (для fairness/мониторинга)."""
    _, test = load_borrowers_split()
    _, meta = load_model()
    test = test.copy()
    test["pd"] = predict_proba(test)
    test["approved"] = (test["pd"] < meta["threshold"]).astype(int)
    return test


def default_application() -> dict:
    """Пример заявки по умолчанию (надёжный заёмщик) для интерфейса."""
    return {
        "age": 36,
        "gender": "Мужской",
        "region": "Москва и МО",
        "income": 180_000,
        "pdn": 28.0,
        "credit_score": 740,
        "num_existing_loans": 0,
        "employment_type": "Зарплатный клиент",
        "employment_months": 60,
        "has_salary_account": 1,
        "program": "Семейная",
        "loan_amount": 6_000_000,
        "ltv": 70.0,
        "down_payment_pct": 30.0,
        "term_years": 20,
    }
