"""
Инференс двухступенчатой модели автоматической оценки недвижимости (AVM).

Этап 1 — CatBoost (базовая оценка), этап 2 — LightGBM (финальная оценка
с учётом прогноза этапа 1). Архитектура соответствует описанию сервиса
«Домклик» в разделе 2.2 ВКР.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import config as C
from .train import AVM_CAT_COLS


@lru_cache(maxsize=1)
def load_model():
    stage1 = CatBoostRegressor()
    stage1.load_model(str(C.AVM_STAGE1_PATH))
    stage2 = joblib.load(C.AVM_STAGE2_PATH)
    meta = json.loads(C.AVM_META_PATH.read_text(encoding="utf-8"))
    return stage1, stage2, meta


def predict_one(values: dict) -> dict:
    """Оценка стоимости одного объекта. Возвращает итог и вклад каждого этапа."""
    stage1, stage2, meta = load_model()
    feats = meta["features"]
    row = pd.DataFrame([values])[feats]

    # Этап 1 (CatBoost) — в логарифме цены
    log_p1 = float(stage1.predict(Pool(row, cat_features=meta["cat_idx"]))[0])
    price_stage1 = float(np.exp(log_p1))

    # Этап 2 (LightGBM) поверх признаков + прогноз этапа 1
    X2 = row.copy()
    for col in AVM_CAT_COLS:
        X2[col] = pd.Categorical(X2[col], categories=meta["cat_levels"][col])
    X2["stage1_pred"] = log_p1
    log_final = float(stage2.predict(X2)[0])
    price_final = float(np.exp(log_final))

    err = meta["metrics"]["mape_final"]
    return {
        "price_stage1": price_stage1,
        "price_final": price_final,
        "price_per_m2": price_final / values["area"],
        "mape": err,
        "low": price_final * (1 - err),   # доверительный диапазон ± MAPE
        "high": price_final * (1 + err),
    }


def default_property() -> dict:
    """Пример объекта по умолчанию для интерфейса."""
    return {
        "area": 56.0,
        "rooms": 2,
        "floor": 7,
        "total_floors": 17,
        "year_built": 2015,
        "distance_to_center": 9.0,
        "metro_minutes": 10,
        "region": "Москва и МО",
        "wall_material": "Монолит",
        "renovation": "Типовой ремонт",
        "has_balcony": 1,
    }
