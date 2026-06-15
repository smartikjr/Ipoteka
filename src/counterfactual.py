"""
Контрфактические рекомендации: «что изменить, чтобы заявку одобрили».

Для отклонённой заявки перебирает по одному изменению ключевых параметров
и находит минимальное значение, при котором вероятность дефолта (PD) опускается
ниже порога одобрения. Это прикладная реализация контрфактической интерпретации,
упомянутой в разделе 3.2 ВКР (наряду со SHAP — часть Рекомендации 3).
"""

from __future__ import annotations

import numpy as np

from . import scoring


def _pd(values: dict) -> float:
    return scoring.predict_one(values)["pd"]


def suggest(values: dict, max_items: int = 4) -> list[dict]:
    """Список выполнимых одиночных изменений, ведущих к одобрению."""
    base = scoring.predict_one(values)
    thr = base["threshold"]
    if base["approved"]:
        return []

    property_value = values["loan_amount"] / (1 - values["down_payment_pct"] / 100)
    suggestions: list[dict] = []

    # 1. Увеличить первоначальный взнос (снизить LTV)
    for dp in np.arange(values["down_payment_pct"] + 5, 61, 5):
        loan = property_value * (1 - dp / 100)
        cand = {**values, "down_payment_pct": float(dp), "ltv": float(100 - dp),
                "loan_amount": float(loan)}
        if _pd(cand) < thr:
            suggestions.append({
                "label": "Увеличить первоначальный взнос",
                "action": f"до {dp:.0f}% (взнос {property_value * dp / 100:,.0f} ₽)".replace(",", " "),
                "new_pd": _pd(cand)})
            break

    # 2. Снизить долговую нагрузку (ПДН)
    for pdn in np.arange(values["pdn"] - 5, 9, -5):
        cand = {**values, "pdn": float(pdn)}
        if _pd(cand) < thr:
            suggestions.append({"label": "Снизить долговую нагрузку (ПДН)",
                                "action": f"до {pdn:.0f}%", "new_pd": _pd(cand)})
            break

    # 3. Улучшить кредитную историю
    for cs in range(int(values["credit_score"]) + 30, 851, 30):
        cand = {**values, "credit_score": int(cs)}
        if _pd(cand) < thr:
            suggestions.append({"label": "Повысить балл кредитной истории",
                                "action": f"до {cs}", "new_pd": _pd(cand)})
            break

    # 4. Закрыть часть действующих кредитов
    if values["num_existing_loans"] > 0:
        for k in range(int(values["num_existing_loans"]) - 1, -1, -1):
            cand = {**values, "num_existing_loans": int(k)}
            if _pd(cand) < thr:
                suggestions.append({"label": "Закрыть действующие кредиты",
                                    "action": f"оставить {k} шт.", "new_pd": _pd(cand)})
                break

    # 5. Уменьшить сумму кредита (более доступный объект)
    for factor in (0.9, 0.8, 0.7, 0.6, 0.5):
        loan = values["loan_amount"] * factor
        cand = {**values, "loan_amount": float(loan)}
        if _pd(cand) < thr:
            suggestions.append({
                "label": "Снизить сумму кредита",
                "action": f"до {loan:,.0f} ₽ (−{(1 - factor) * 100:.0f}%)".replace(",", " "),
                "new_pd": _pd(cand)})
            break

    # 6. Стать зарплатным клиентом банка
    if not values.get("has_salary_account"):
        cand = {**values, "has_salary_account": 1}
        if _pd(cand) < thr:
            suggestions.append({"label": "Стать зарплатным клиентом банка",
                                "action": "перевести зарплату в банк", "new_pd": _pd(cand)})

    # Запасной вариант: комплекс мер, если одиночных изменений недостаточно
    if len(suggestions) < 2:
        dp = min(values["down_payment_pct"] + 20, 60)
        loan = property_value * (1 - dp / 100) * 0.8
        combo = {**values, "down_payment_pct": float(dp), "ltv": float(100 - dp),
                 "loan_amount": float(loan), "pdn": max(values["pdn"] - 15, 10.0),
                 "num_existing_loans": 0, "has_salary_account": 1}
        if _pd(combo) < thr:
            suggestions.append({
                "label": "Комплекс мер",
                "action": f"взнос до {dp:.0f}%, сумма −20%, закрыть кредиты, "
                          "перейти на зарплатный проект",
                "new_pd": _pd(combo)})

    suggestions.sort(key=lambda s: s["new_pd"])
    return suggestions[:max_items]
