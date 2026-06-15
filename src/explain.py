"""
Интерпретируемость решений модели скоринга методом SHAP.

Реализует Рекомендацию 3 из ВКР: объяснение причин решения по ипотечной заявке
для заёмщика и регулятора (SHapley Additive exPlanations). Для каждого фактора
рассчитывается его вклад в итоговую оценку риска (в логарифме шансов).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from . import config as C
from .scoring import load_model, prepare_frame


@lru_cache(maxsize=1)
def _explainer():
    model, _ = load_model()
    return shap.TreeExplainer(model)


def explain_one(values: dict) -> dict:
    """
    Вернуть вклад каждого признака (SHAP) для одной заявки.

    Положительный вклад -> повышает оценку риска (в сторону отказа),
    отрицательный -> снижает риск (в сторону одобрения).
    """
    model, meta = load_model()
    explainer = _explainer()
    X = prepare_frame(pd.DataFrame([values]), meta)

    sv = explainer.shap_values(X)
    # Для бинарного LightGBM shap может вернуть list[класс0, класс1] или массив
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:           # (n, n_features, n_classes)
        sv = sv[:, :, -1]
    contrib = sv[0]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[-1])

    items = []
    for feat, val in zip(meta["features"], contrib):
        items.append(
            {
                "feature": feat,
                "label": C.FEATURE_LABELS_RU.get(feat, feat),
                "value": values[feat],
                "shap": float(val),
            }
        )
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return {"base_value": float(expected), "contributions": items}


def top_reasons(values: dict, k: int = 4) -> list[str]:
    """Сформулировать главные причины отказа (факторы, повышающие риск)."""
    expl = explain_one(values)
    risk_factors = [c for c in expl["contributions"] if c["shap"] > 0][:k]
    reasons = []
    for c in risk_factors:
        reasons.append(f"{c['label']}: {_fmt_value(c['feature'], c['value'])}")
    return reasons


def _fmt_value(feature: str, value) -> str:
    if feature in ("income", "loan_amount"):
        return f"{value:,.0f} руб.".replace(",", " ")
    if feature in ("pdn", "ltv", "down_payment_pct"):
        return f"{value:g}%"
    return f"{value}"
