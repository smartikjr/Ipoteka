"""
Генерация статичных графиков (PNG) для презентации к защите ВКР.

Создаёт в docs/figures/ ключевые визуализации прототипа «мини-Домклик»:
SHAP-объяснение, двухступенчатую AVM, fairness, мониторинг (PSI), карту рисков,
эффекты ИИ, ROC-кривую скоринга и точность AVM.

Запуск:  python -m scripts.make_figures
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import Pool
from sklearn.metrics import roc_auc_score, roc_curve

from src import avm, config as C, explain, fairness, monitoring, risk_map, scoring
from src.train import AVM_CAT_COLS, load_borrowers_split

GREEN, RED, ORANGE, GREY, DARK = "#21A038", "#E23C3C", "#F2A33A", "#8A9A90", "#15281C"
OUT = C.ROOT_DIR / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DejaVu Sans, Arial", size=15, color=DARK),
    margin=dict(l=60, r=30, t=70, b=50),
    title=dict(font=dict(size=19)),
)

# Легенда снизу — чтобы не пересекаться с заголовком в статичных PNG
BOTTOM_LEGEND = dict(orientation="h", y=-0.16, x=0.5, xanchor="center")


def save(fig, name: str, w: int = 1000, h: int = 600, **layout_over):
    layout = dict(BASE_LAYOUT)
    layout.update(layout_over)
    fig.update_layout(**layout)
    path = OUT / name
    fig.write_image(str(path), width=w, height=h, scale=2)
    print(f"  ✓ {name}")


def fig_shap():
    risky = dict(
        scoring.default_application(),
        pdn=64.0, credit_score=560, income=58_000, num_existing_loans=4,
        has_salary_account=0, employment_type="ИП", down_payment_pct=15.0,
        program="Рыночная", age=27,
    )
    res = scoring.predict_one(risky)
    expl = explain.explain_one(risky)
    items = expl["contributions"][:9][::-1]
    vals = [it["shap"] for it in items]
    fig = go.Figure(go.Bar(
        x=vals, y=[it["label"] for it in items], orientation="h",
        marker_color=[RED if v > 0 else GREEN for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP-объяснение решения: {res['verdict']} (PD={res['pd']:.0%})",
        xaxis_title="Вклад фактора в оценку риска (лог-шансы)",
        xaxis_range=[-0.5, max(vals) * 1.22],
    )
    save(fig, "01_scoring_shap.png", h=560, margin=dict(l=215, r=90, t=70, b=50))


def fig_avm_stages():
    res = avm.predict_one(avm.default_property())
    fig = go.Figure(go.Bar(
        x=["Этап 1 (CatBoost)", "Этап 2 (LightGBM, финал)"],
        y=[res["price_stage1"], res["price_final"]],
        marker_color=[GREY, GREEN],
        text=[f"{res['price_stage1']:,.0f} ₽".replace(",", " "),
              f"{res['price_final']:,.0f} ₽".replace(",", " ")],
        textposition="outside",
    ))
    fig.update_layout(title=f"Двухступенчатая оценка недвижимости (AVM), MAPE={res['mape']:.1%}",
                      yaxis_title="Стоимость, ₽")
    save(fig, "02_avm_stages.png", h=520)


def fig_fairness():
    t = fairness.group_metrics("age_group")
    s = fairness.summary("age_group")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Доля одобрений (DP)", x=t["Группа"],
                         y=t["Доля одобрений"], marker_color=GREEN))
    fig.add_trace(go.Bar(name="Одобрение «хороших» (EO)", x=t["Группа"],
                         y=t["Одобрение «хороших»"], marker_color="#9FD8B4"))
    fig.update_layout(
        title=f"Справедливость по возрасту · правило 4/5: {s['dp_ratio']:.2f} "
              f"({'пройдено' if s['passes_4_5'] else 'нарушение'})",
        barmode="group", yaxis_tickformat=".0%", yaxis_title="Доля одобрений",
        legend=BOTTOM_LEGEND,
    )
    save(fig, "03_fairness.png", margin=dict(l=70, r=30, t=70, b=80))


def fig_monitoring():
    r = monitoring.monitoring_report()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="До шока (обучение)", x=r["bin_labels"],
                         y=r["base_share"], marker_color=GREEN))
    fig.add_trace(go.Bar(name="После шока (текущий период)", x=r["bin_labels"],
                         y=r["shock_share"], marker_color=RED))
    fig.update_layout(
        title=f"Мониторинг (MOC): PSI = {r['psi_score']:.2f} — значимый сдвиг распределения",
        barmode="group", yaxis_tickformat=".0%",
        xaxis_title="Бин скорингового балла (1 — наибольший риск, 10 — наименьший)",
        yaxis_title="Доля популяции",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=70, b=120),
    )
    save(fig, "04_monitoring.png")


def fig_risk_map():
    df = risk_map.risk_frame()
    fig = go.Figure(go.Scatter(
        x=df["probability"], y=df["impact"], mode="markers+text",
        text=df["Риск"], textposition="top center",
        marker=dict(size=42, color=[RED, RED, RED, ORANGE, ORANGE, GREY],
                    line=dict(width=1, color="#fff")),
    ))
    fig.add_hline(y=3.0, line_dash="dot", line_color=GREY)
    fig.add_vline(x=3.0, line_dash="dot", line_color=GREY)
    fig.update_layout(
        title="Карта рисков применения ИИ в ипотечном кредитовании",
        xaxis=dict(title="Вероятность", range=[2.5, 5]),
        yaxis=dict(title="Влияние на банк", range=[2.5, 5]),
    )
    save(fig, "05_risk_map.png", h=620)


def fig_effects():
    f = C.THESIS_FACTS
    labels = ["Автоодобрения, %", "Gini скоринга, %", "Просрочка 90+, %", "Расходы на оценку, %"]
    before = [f["auto_approval_before"], 55, 1.5, 100]
    after = [f["auto_approval_after"], 77.5, 0.7, 70]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="До ИИ", x=labels, y=before, marker_color=GREY,
                         text=before, textposition="outside"))
    fig.add_trace(go.Bar(name="После ИИ", x=labels, y=after, marker_color=GREEN,
                         text=after, textposition="outside"))
    fig.update_layout(title="Эффекты внедрения ИИ в ипотечное кредитование (до / после)",
                      barmode="group", yaxis_title="Значение показателя",
                      legend=BOTTOM_LEGEND)
    save(fig, "06_effects.png", margin=dict(l=60, r=30, t=70, b=80))


def fig_roc():
    _, test = load_borrowers_split()
    proba = scoring.predict_proba(test)
    auc = roc_auc_score(test["default"], proba)
    fpr, tpr, _ = roc_curve(test["default"], proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=GREEN, width=3),
                             name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=GREY, dash="dash"), name="Случайная модель"))
    fig.update_layout(title=f"ROC-кривая модели скоринга · Gini = {2*auc-1:.3f}",
                      xaxis_title="Доля ложных тревог (FPR)",
                      yaxis_title="Доля верных обнаружений (TPR)",
                      legend=dict(x=0.55, y=0.1))
    save(fig, "07_scoring_roc.png", w=720, h=620)


def fig_avm_accuracy():
    stage1, stage2, meta = avm.load_model()
    df = pd.read_csv(C.PROPERTIES_CSV)
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(df, test_size=0.25, random_state=C.RANDOM_SEED)
    feats = meta["features"]
    log1 = stage1.predict(Pool(test[feats], cat_features=meta["cat_idx"]))
    X2 = test[feats].copy()
    for col in AVM_CAT_COLS:
        X2[col] = pd.Categorical(X2[col], categories=meta["cat_levels"][col])
    X2["stage1_pred"] = log1
    pred = np.exp(stage2.predict(X2))
    actual = test["price"].to_numpy()
    mape = float(np.mean(np.abs((actual - pred) / actual)))
    s = test.sample(min(2000, len(test)), random_state=1).index
    idx = test.index.get_indexer(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual[idx] / 1e6, y=pred[idx] / 1e6, mode="markers",
                             marker=dict(color=GREEN, size=5, opacity=0.4), name="Объекты"))
    lim = max(actual.max(), pred.max()) / 1e6
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color=RED, dash="dash"), name="Идеальная оценка"))
    fig.update_layout(title=f"Точность AVM: прогноз vs факт · MAPE = {mape:.1%}",
                      xaxis_title="Фактическая цена, млн ₽",
                      yaxis_title="Оценка модели, млн ₽",
                      legend=dict(x=0.02, y=0.98))
    save(fig, "08_avm_accuracy.png", w=720, h=620)


def main():
    print("Генерация графиков в docs/figures/ ...")
    fig_shap()
    fig_avm_stages()
    fig_fairness()
    fig_monitoring()
    fig_risk_map()
    fig_effects()
    fig_roc()
    fig_avm_accuracy()
    print("Готово.")


if __name__ == "__main__":
    main()
