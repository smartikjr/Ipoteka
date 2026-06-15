"""
Портфельная аналитика и стресс-тестирование.

Реализует третий технологический узел из раздела 2.2 ВКР — прогнозирование
поведения портфеля и расчёт ожидаемых кредитных потерь (МСФО 9):

    EL = PD × LGD × EAD

- EAD (Exposure at Default) — сумма кредита (остаток ссудной задолженности);
- PD (Probability of Default) — оценка скоринговой модели;
- LGD (Loss Given Default) — доля потерь с учётом стоимости залога: при падении
  цен на недвижимость стоимость обеспечения снижается и LGD растёт.

Стресс-тест применяет к портфелю макрошок (рост ключевой ставки, падение доходов,
снижение цен на жильё), пересчитывает PD моделью и показывает рост ожидаемых потерь.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C, scoring

PRICE_HAIRCUT = 0.15   # дисконт вынужденной реализации залога
LGD_FLOOR = 0.08       # минимальная доля потерь (издержки, время реализации)


@lru_cache(maxsize=1)
def scored_portfolio() -> pd.DataFrame:
    """Полный набор заёмщиков с рассчитанной PD."""
    df = pd.read_csv(C.BORROWERS_CSV).copy()
    df["pd"] = scoring.predict_proba(df)
    return df


def booked_loans() -> pd.DataFrame:
    """«Выданные» кредиты портфеля — одобренные заявки (PD ниже порога)."""
    df = scored_portfolio()
    return df[df["pd"] < C.PD_APPROVE_THRESHOLD].reset_index(drop=True)


def lgd(df: pd.DataFrame, price_shock: float = 0.0) -> np.ndarray:
    """LGD с учётом стоимости залога после шока цен на жильё."""
    collateral = df["property_value"].to_numpy() * (1 - price_shock) * (1 - PRICE_HAIRCUT)
    recovery = np.clip(collateral / df["loan_amount"].to_numpy(), 0.0, 1.0)
    return np.clip(1 - recovery, LGD_FLOOR, 1.0)


def expected_loss(df: pd.DataFrame, pd_col: str = "pd", price_shock: float = 0.0) -> np.ndarray:
    ead = df["loan_amount"].to_numpy()
    return df[pd_col].to_numpy() * lgd(df, price_shock) * ead


def summary(df: pd.DataFrame, pd_col: str = "pd", price_shock: float = 0.0) -> dict:
    ead = df["loan_amount"].to_numpy()
    el = expected_loss(df, pd_col, price_shock)
    return {
        "n": int(len(df)),
        "ead_total": float(ead.sum()),
        "mean_pd": float(df[pd_col].mean()),
        "lgd_mean": float(lgd(df, price_shock).mean()),
        "el_total": float(el.sum()),
        "el_pct": float(el.sum() / ead.sum()) if ead.sum() else 0.0,
        "npl_share": float((df[pd_col] > C.PD_APPROVE_THRESHOLD).mean()),
    }


def apply_macro(df: pd.DataFrame, rate_delta_pp: float, income_shock_pct: float,
                price_drop_pct: float) -> pd.DataFrame:
    """Применить макрошок и пересчитать PD моделью."""
    s = df.copy()
    # Рост ключевой ставки -> рост долговой нагрузки (дороже обслуживание)
    s["pdn"] = np.clip(s["pdn"] + rate_delta_pp * 1.2, 3, 95)
    # Падение реальных доходов
    s["income"] = s["income"] * (1 - income_shock_pct / 100)
    # Падение цен на жильё -> рост LTV
    s["ltv"] = np.clip(s["ltv"] / max(1 - price_drop_pct / 100, 0.5), 30, 130)
    s["pd_stressed"] = scoring.predict_proba(s)
    return s


def stress_test(rate_delta_pp: float, income_shock_pct: float,
                price_drop_pct: float) -> dict:
    """Сравнение базового и стрессового сценариев по портфелю выданных кредитов."""
    base = booked_loans()
    base_sum = summary(base, "pd", 0.0)

    stressed = apply_macro(base, rate_delta_pp, income_shock_pct, price_drop_pct)
    price_shock = price_drop_pct / 100
    stress_sum = summary(stressed, "pd_stressed", price_shock)

    return {
        "base": base_sum,
        "stress": stress_sum,
        "el_add": stress_sum["el_total"] - base_sum["el_total"],
        "pd_delta": stress_sum["mean_pd"] - base_sum["mean_pd"],
    }


# --------------------------------------------------------------------------- #
# Структура портфеля и кривая дефолтов
# --------------------------------------------------------------------------- #
def structure(by: str) -> pd.DataFrame:
    """Доля портфеля (по сумме кредитов) в разрезе признака."""
    df = booked_loans()
    g = df.groupby(by, observed=True)["loan_amount"].agg(["sum", "count"]).reset_index()
    g["share"] = g["sum"] / g["sum"].sum()
    g = g.rename(columns={by: "Категория", "sum": "Объём", "count": "Кредитов"})
    return g.sort_values("Объём", ascending=False).reset_index(drop=True)


def vintage_default_curve(months: int = 120) -> pd.DataFrame:
    """
    Маржинальная вероятность дефолта по месяцам с момента выдачи («горб дефолтов»).

    Соответствует наблюдению из раздела 2.1 ВКР: вероятность дефолта по ипотечному
    кредиту достигает максимума на 24–36 месяце. Кривая смоделирована аналитически
    (лог-нормальная форма хазард-функции) для иллюстрации.
    """
    t = np.arange(1, months + 1)
    peak = 30.0
    shape = np.exp(-((np.log(t) - np.log(peak)) ** 2) / (2 * 0.45 ** 2))
    shape = shape / shape.max()
    target_cumulative = 0.045          # реалистичная накопленная дефолтность ~4.5% за срок
    hazard = shape * (target_cumulative / shape.sum())
    cum_default = 1 - np.cumprod(1 - hazard)
    return pd.DataFrame({"Месяц": t, "Маржинальная PD": hazard,
                         "Накопленная дефолтность": cum_default})
