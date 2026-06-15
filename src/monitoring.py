"""
Мониторинг ИИ-моделей (Model Operations Center, MOC).

Реализует Рекомендации 1 и 4 из ВКР: непрерывный контроль стабильности модели
и раннее выявление её деградации при изменении макросреды.

Ключевая метрика — Population Stability Index (PSI), показывающий сдвиг
распределения относительно периода обучения:
  PSI < 0.10  — модель стабильна;
  0.10–0.25   — умеренный сдвиг (наблюдение);
  > 0.25      — значимый сдвиг (требуется ретрейнинг модели).

Сценарий «шок 2024–2025» моделирует резкий рост ключевой ставки и ухудшение
платёжеспособности заёмщиков — эпизод, описанный в разделах 2.1–2.2 ВКР
(рост просроченной задолженности на 90% за квартал).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import get_scored_test, predict_proba

PSI_WARN = 0.10
PSI_ALERT = 0.25


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI между базовым и текущим распределениями (биннинг по квантилям базы)."""
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)
    b = np.clip(b_counts / b_counts.sum(), 1e-6, None)
    c = np.clip(c_counts / c_counts.sum(), 1e-6, None)
    return float(np.sum((c - b) * np.log(c / b)))


def shocked_frame() -> pd.DataFrame:
    """Отложенная выборка в условиях макрошока (ухудшение платёжеспособности)."""
    df = get_scored_test().copy()
    df["pdn"] = np.clip(df["pdn"] + 15, 3, 80)          # рост долговой нагрузки
    df["income"] = (df["income"] * 0.82).round(-2)       # падение реальных доходов
    df["ltv"] = np.clip(df["ltv"] + 7, 30, 95)           # рост LTV
    df["pd"] = predict_proba(df)
    return df


def status(psi: float) -> tuple[str, str]:
    """Текстовый статус и цвет для значения PSI."""
    if psi < PSI_WARN:
        return "Стабильно", "green"
    if psi < PSI_ALERT:
        return "Умеренный сдвиг — наблюдение", "orange"
    return "Значимый сдвиг — требуется ретрейнинг", "red"


def psi_distribution(baseline: np.ndarray, current: np.ndarray, bins: int = 10):
    """Доли популяции по бинам базового распределения (для PSI-диаграммы)."""
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)
    b_share = b_counts / b_counts.sum()
    c_share = c_counts / c_counts.sum()
    labels = [str(i + 1) for i in range(bins)]  # бины по возрастанию балла (1 — риск выше)
    return labels, b_share, c_share


def monitoring_report() -> dict:
    """Сводка мониторинга: PSI по баллу и по ключевому признаку (ПДН)."""
    base = get_scored_test()
    shock = shocked_frame()

    base_score = 1000 * (1 - base["pd"].to_numpy())
    shock_score = 1000 * (1 - shock["pd"].to_numpy())

    psi_score = compute_psi(base_score, shock_score)
    psi_pdn = compute_psi(base["pdn"].to_numpy(), shock["pdn"].to_numpy())
    bin_labels, base_share, shock_share = psi_distribution(base_score, shock_score)

    return {
        "psi_score": psi_score,
        "psi_pdn": psi_pdn,
        "status_score": status(psi_score),
        "approval_base": float(base["approved"].mean()),
        "approval_shock": float((shock["pd"] < 0.10).mean()),
        "mean_pd_base": float(base["pd"].mean()),
        "mean_pd_shock": float(shock["pd"].mean()),
        "base_score": base_score,
        "shock_score": shock_score,
        "bin_labels": bin_labels,
        "base_share": base_share,
        "shock_share": shock_share,
    }
