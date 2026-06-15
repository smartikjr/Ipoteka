"""
Проверка моделей скоринга на справедливость (fairness).

Реализует Рекомендацию 2 из ВКР: тестирование ИИ-модели на отсутствие
алгоритмической предвзятости по защищённым признакам (пол, возраст, регион)
с использованием стандартных метрик справедливости:

  - Demographic Parity (демографический паритет) — равенство долей одобрений;
  - Equal Opportunity (равенство возможностей) — равенство долей одобрений
    среди «хороших» заёмщиков (фактически не допустивших дефолт);
  - Equalized Odds — дополнительно равенство долей ошибочных одобрений
    среди «плохих» заёмщиков.

Также применяется «правило 4/5» (80%): отношение минимальной доли одобрений
к максимальной не должно быть ниже 0.8.
"""

from __future__ import annotations

import pandas as pd

from .scoring import get_scored_test

FOUR_FIFTHS = 0.8  # порог «правила 4/5»


def group_metrics(attr: str) -> pd.DataFrame:
    """Метрики справедливости в разрезе значений защищённого признака."""
    df = get_scored_test()
    rows = []
    for value, g in df.groupby(attr, observed=True):
        good = g[g["default"] == 0]
        bad = g[g["default"] == 1]
        rows.append(
            {
                "Группа": value,
                "Заявок": len(g),
                "Доля одобрений": g["approved"].mean(),               # Demographic Parity
                "Одобрение «хороших»": good["approved"].mean() if len(good) else 0.0,  # Equal Opportunity (TPR)
                "Одобрение «плохих»": bad["approved"].mean() if len(bad) else 0.0,     # FPR
                "Факт. дефолтность": g["default"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("Группа").reset_index(drop=True)


def summary(attr: str) -> dict:
    """Сводные показатели разрыва (gap) и вердикт по «правилу 4/5»."""
    t = group_metrics(attr)
    dp = t["Доля одобрений"]
    eo = t["Одобрение «хороших»"]
    dp_gap = float(dp.max() - dp.min())
    dp_ratio = float(dp.min() / dp.max()) if dp.max() > 0 else 0.0
    eo_gap = float(eo.max() - eo.min())
    return {
        "attr": attr,
        "dp_gap": dp_gap,           # разрыв демографического паритета
        "dp_ratio": dp_ratio,       # отношение (правило 4/5)
        "eo_gap": eo_gap,           # разрыв равенства возможностей
        "passes_4_5": dp_ratio >= FOUR_FIFTHS,
        "worst_group": str(t.loc[dp.idxmin(), "Группа"]),
        "best_group": str(t.loc[dp.idxmax(), "Группа"]),
    }
