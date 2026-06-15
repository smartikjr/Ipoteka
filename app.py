"""
«мини-Домклик» — интерактивная демонстрация применения искусственного интеллекта
в ипотечном кредитовании.

Прототип к выпускной квалификационной работе
«Возможности и риски использования ИИ в ипотечном кредитовании
в российских коммерческих банках (на примере ПАО Сбербанк)».

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

# Гарантируем, что корень проекта в пути импорта (устойчиво к любому cwd)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import (
    assistant,
    avm,
    bootstrap,
    config as C,
    counterfactual,
    explain,
    fairness,
    geo,
    monitoring,
    portfolio,
    report,
    risk_map,
    scoring,
)

# --------------------------------------------------------------------------- #
# Оформление
# --------------------------------------------------------------------------- #
GREEN = "#21A038"
RED = "#E23C3C"
ORANGE = "#F2A33A"
GREY = "#8A9A90"
DARK = "#15281C"

st.set_page_config(
    page_title="мини-Домклик · ИИ в ипотеке",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      html, body, [data-testid="stAppViewContainer"], [class*="css"] {
          font-family: 'Inter', -apple-system, Segoe UI, sans-serif;
      }
      .stApp { background: #F4F7F5; }
      [data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }
      h1, h2, h3 { color: #15281C; letter-spacing: -0.01em; }

      /* Боковая панель */
      [data-testid="stSidebar"] {
          background: #FFFFFF; border-right: 1px solid #E6ECE8;
      }
      [data-testid="stSidebar"] [role="radiogroup"] { gap: 3px; }
      [data-testid="stSidebar"] [role="radiogroup"] label {
          padding: 7px 12px; border-radius: 10px; transition: background .15s; cursor: pointer;
      }
      [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #EAF5EE; }
      [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #EAF5EE; }
      [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #0E7A2A !important; font-weight: 600; }

      /* Карточки (bordered-контейнеры) */
      [data-testid="stVerticalBlockBorderWrapper"] {
          background: #fff; border-radius: 14px; border: 1px solid #E6ECE8 !important;
          box-shadow: 0 2px 12px rgba(20,40,28,0.05);
      }

      /* Метрики */
      [data-testid="stMetric"] {
          background: #fff; border: 1px solid #E6ECE8; border-radius: 12px;
          padding: 12px 16px; box-shadow: 0 1px 6px rgba(20,40,28,0.04);
      }
      [data-testid="stMetricValue"] { font-size: 1.5rem; color: #15281C; }
      [data-testid="stMetricLabel"] { color: #5C6B62; }

      /* Кнопки */
      .stButton > button {
          border-radius: 10px; border: 1px solid #CBE5D4; color: #15281C;
          font-weight: 500; transition: all .15s;
      }
      .stButton > button:hover { border-color: #21A038; color: #0E7A2A; background: #F1F9F3; }
      .stButton > button[kind="primary"], .stDownloadButton > button {
          background: linear-gradient(110deg, #21A038, #16863A); color: #fff; border: none;
      }
      .stDownloadButton > button:hover { filter: brightness(1.05); color:#fff; }

      /* Чат */
      [data-testid="stChatInput"] textarea { border-radius: 12px; }

      /* Hero-баннер */
      .hero {
          background: linear-gradient(115deg, #21A038 0%, #0E7A2A 100%);
          border-radius: 16px; padding: 22px 28px; margin-bottom: 16px;
          box-shadow: 0 8px 24px rgba(33,160,56,0.22);
      }
      .hero h1 { color: #fff; margin: 0; font-size: 1.6rem; font-weight: 700; }
      .hero p { color: #E9F8EF; margin: 7px 0 0; font-size: 0.97rem; line-height: 1.4; }

      .pill { display:inline-block; padding:3px 11px; border-radius:12px;
              font-size:0.78rem; font-weight:600; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #
def fmt_rub(x: float) -> str:
    return f"{x:,.0f} ₽".replace(",", " ")


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


def hero(title: str, subtitle: str = "") -> None:
    """Градиентная шапка страницы."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


@st.cache_resource
def _warmup():
    """Подготовка артефактов при первом запуске и прогрев моделей."""
    if not bootstrap.artifacts_ready():
        with st.spinner("Первый запуск: генерация данных и обучение моделей (~1–2 мин)…"):
            bootstrap.ensure_artifacts(log=lambda m: None)
    scoring.load_model()
    avm.load_model()
    return True


@st.cache_data
def scoring_meta() -> dict:
    return scoring.load_model()[1]


@st.cache_data
def avm_meta() -> dict:
    return avm.load_model()[2]


_warmup()


# --------------------------------------------------------------------------- #
# Боковая панель
# --------------------------------------------------------------------------- #
PAGES = [
    "🏠 Обзор",
    "📝 Скоринг и объяснение (SHAP)",
    "🤖 ИИ-консультант",
    "🏢 Оценка недвижимости (AVM)",
    "🗺️ Карта недвижимости",
    "📊 Аналитика портфеля",
    "🌪️ Стресс-тест портфеля",
    "⚖️ Справедливость моделей",
    "📉 Мониторинг моделей (MOC)",
    "⚠️ Карта рисков",
    "📈 Эффекты внедрения ИИ",
]

with st.sidebar:
    st.markdown("### 🏦 мини-Домклик")
    st.caption("ИИ в ипотечном кредитовании · прототип к ВКР")
    page = st.radio("Раздел", PAGES, label_visibility="collapsed", key="nav")

    st.divider()
    sm = scoring_meta()["metrics"]
    am = avm_meta()["metrics"]
    st.markdown("**Качество моделей**")
    st.metric("Gini скоринга", f"{sm['gini']:.3f}", help="Целевой ориентир ВКР: 0.75–0.80")
    st.metric("Погрешность AVM (MAPE)", f"{am['mape_final']:.1%}", help="Ориентир ВКР: 4–5%")

    st.divider()
    st.caption(
        "⚠️ Данные синтетические (сгенерированы программно) и калиброваны под "
        "параметры ВКР. Это демонстрация методологии, а не реальные данные банка."
    )


# =========================================================================== #
# СТРАНИЦА: ОБЗОР
# =========================================================================== #
def page_overview():
    hero(
        "🏦 мини-Домклик — ИИ в ипотечном кредитовании",
        "Интерактивный прототип к ВКР «Возможности и риски использования ИИ в ипотечном "
        "кредитовании в российских коммерческих банках (на примере ПАО Сбербанк)». "
        "Четыре узла ипотечного ИИ-конвейера и рекомендации третьей главы — в действии.",
    )

    f = C.THESIS_FACTS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ипотечный портфель Сбербанка, 2025", f"{f['portfolio_2025_trln']} трлн ₽",
              f"+{f['portfolio_growth_pct']}%")
    c2.metric("Доля банка на рынке ипотеки", f"{f['market_share_pct']}%")
    c3.metric("Автоодобрения в «Домклик»",
              f"{f['auto_approval_after']}%", f"+{f['auto_approval_after']-f['auto_approval_before']} п.п.")
    c4.metric("Эффект ИИ (группа Сбербанк)", f"≈ {f['ai_effect_bln']} млрд ₽/год")

    st.divider()
    st.subheader("Что демонстрирует прототип")
    cols = st.columns(2)
    cards = [
        ("📝 Кредитный скоринг + SHAP",
         "Оценка вероятности дефолта (PD) ансамблем градиентного бустинга "
         f"(Gini ≈ {scoring_meta()['metrics']['gini']:.2f}) и объяснение каждого решения "
         "методом SHAP. **Рекомендация 3** (интерпретируемость)."),
        ("🏢 Автооценка недвижимости (AVM)",
         "Двухступенчатая модель CatBoost → LightGBM, погрешность "
         f"≈ {avm_meta()['metrics']['mape_final']:.1%}, как у сервиса «Домклик» (раздел 2.2)."),
        ("⚖️ Тестирование на справедливость",
         "Метрики Demographic Parity, Equal Opportunity и «правило 4/5» по полу, "
         "возрасту, региону. **Рекомендация 2** (противодействие предвзятости)."),
        ("📊 Мониторинг моделей (MOC) + карта рисков",
         "Контроль стабильности модели (PSI) при макрошоке и карта рисков из главы 3. "
         "**Рекомендации 1 и 4** (ретрейнинг, мониторинг)."),
    ]
    for i, (title, text) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(f"#### {title}")
            st.write(text)

    st.divider()
    st.subheader("Архитектура ИИ-конвейера ипотеки (по разделу 2.2 ВКР)")
    st.markdown(
        """
        | Узел | Модель | Реализация в прототипе |
        |---|---|---|
        | 1. Оценка залога (AVM) | CatBoost + LightGBM | Раздел «Оценка недвижимости» |
        | 2. Кредитный скоринг заёмщика | Градиентный бустинг (LightGBM) | Раздел «Скоринг и объяснение» |
        | 3. Прогноз поведения портфеля | Мониторинг PD / PSI | Раздел «Мониторинг моделей» |
        | 4. Интерпретация и контроль | SHAP, fairness, карта рисков | Соответствующие разделы |
        """
    )
    st.info(
        "Навигация — слева. Рекомендуемый порядок для защиты: Скоринг → SHAP → "
        "AVM → Справедливость → Мониторинг → Карта рисков → Эффекты ИИ."
    )


# =========================================================================== #
# СТРАНИЦА: СКОРИНГ + SHAP
# =========================================================================== #
def page_scoring():
    hero(
        "📝 Кредитный скоринг заёмщика и объяснение решения",
        "Модель оценивает вероятность дефолта (PD) и принимает решение. "
        "Защищённые признаки (пол) в модель не подаются.",
    )

    presets = {
        "Надёжный заёмщик": scoring.default_application(),
        "Рискованный заёмщик": dict(
            scoring.default_application(),
            pdn=64.0, credit_score=560, income=58_000, num_existing_loans=4,
            has_salary_account=0, employment_type="ИП", down_payment_pct=15.0,
            program="Рыночная", age=27,
        ),
        "Пограничный случай": dict(
            scoring.default_application(),
            pdn=45.0, credit_score=650, income=95_000, num_existing_loans=2,
            down_payment_pct=20.0, program="Рыночная",
        ),
    }
    preset_name = st.selectbox("Готовый пример заявки", list(presets.keys()))
    p = presets[preset_name]

    with st.form("scoring_form"):
        st.markdown("**Параметры заявки**")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Возраст, лет", 21, 70, int(p["age"]))
            gender = st.selectbox("Пол (для fairness, не в модели)", C.GENDERS,
                                  index=C.GENDERS.index(p["gender"]))
            region = st.selectbox("Регион", C.REGIONS, index=C.REGIONS.index(p["region"]))
            employment_type = st.selectbox("Тип занятости", C.EMPLOYMENT_TYPES,
                                           index=C.EMPLOYMENT_TYPES.index(p["employment_type"]))
            has_salary_account = st.checkbox("Зарплатный клиент банка",
                                             value=bool(p["has_salary_account"]))
        with col2:
            income = st.number_input("Доход, ₽/мес.", 30_000, 1_500_000,
                                     int(p["income"]), step=5_000)
            pdn = st.slider("ПДН (долговая нагрузка), %", 3.0, 80.0, float(p["pdn"]))
            credit_score = st.slider("Балл кредитной истории", 300, 850, int(p["credit_score"]))
            num_existing_loans = st.slider("Действующих кредитов, шт.", 0, 6,
                                           int(p["num_existing_loans"]))
            employment_months = st.number_input("Стаж на текущем месте, мес.", 1, 360,
                                                 int(p["employment_months"]))
        with col3:
            program = st.selectbox("Программа", C.PROGRAMS, index=C.PROGRAMS.index(p["program"]))
            property_value = st.number_input("Стоимость недвижимости, ₽", 1_500_000, 60_000_000,
                                             6_000_000, step=100_000)
            down_payment_pct = st.slider("Первоначальный взнос, %", 10.0, 60.0,
                                         float(p["down_payment_pct"]))
            term_years = st.selectbox("Срок кредита, лет", [10, 15, 20, 25, 30],
                                      index=[10, 15, 20, 25, 30].index(int(p["term_years"])))
            loan_amount = property_value * (1 - down_payment_pct / 100)
            ltv = 100 - down_payment_pct
            st.caption(f"Сумма кредита: **{fmt_rub(loan_amount)}** · LTV ≈ **{ltv:.0f}%**")

        submitted = st.form_submit_button("Рассчитать решение", type="primary",
                                          width="stretch")

    values = dict(
        age=age, gender=gender, region=region, income=income, pdn=pdn,
        credit_score=credit_score, num_existing_loans=num_existing_loans,
        employment_type=employment_type, employment_months=employment_months,
        has_salary_account=int(has_salary_account), program=program,
        loan_amount=loan_amount, ltv=ltv, down_payment_pct=down_payment_pct,
        term_years=term_years,
    )

    if submitted or True:  # показываем результат сразу для выбранного пресета
        res = scoring.predict_one(values)
        # Сохраняем заявку для ИИ-консультанта (раздел «оцени мою заявку»)
        st.session_state["last_app"] = values
        st.session_state["last_result"] = res
        st.divider()
        left, right = st.columns([1, 1.3])
        with left:
            color = GREEN if res["approved"] else RED
            st.markdown(
                f"### Решение: {pill(res['verdict'], color)}", unsafe_allow_html=True
            )
            m1, m2 = st.columns(2)
            m1.metric("Вероятность дефолта (PD)", f"{res['pd']:.1%}")
            m2.metric("Скоринговый балл", f"{res['score']} / 1000")
            st.caption(
                f"Порог одобрения: PD < {res['threshold']:.0%}. "
                "Решение принято автоматически за доли секунды."
            )
            # Датчик PD
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["pd"] * 100,
                number={"suffix": "%", "font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, res["threshold"] * 100], "color": "#DDF1E2"},
                        {"range": [res["threshold"] * 100, 100], "color": "#FBE3E3"},
                    ],
                    "threshold": {"line": {"color": DARK, "width": 3},
                                  "value": res["threshold"] * 100},
                },
                title={"text": "PD, %"},
            ))
            gauge.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge, width="stretch")

        with right:
            st.markdown("#### Объяснение решения (SHAP)")
            expl = explain.explain_one(values)
            items = expl["contributions"][:9][::-1]
            labels = [it["label"] for it in items]
            vals = [it["shap"] for it in items]
            colors = [RED if v > 0 else GREEN for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>вклад в риск: %{x:+.3f}<extra></extra>",
            ))
            fig.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Вклад в оценку риска (лог-шансы)",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"{pill('красный', RED)} — повышает риск (в сторону отказа), "
                f"{pill('зелёный', GREEN)} — снижает риск (в сторону одобрения).",
                unsafe_allow_html=True,
            )

        if not res["approved"]:
            reasons = explain.top_reasons(values)
            st.error("**Основные причины отказа** (для заёмщика, ст. 16 ФЗ № 152-ФЗ):\n\n"
                     + "\n".join(f"- {r}" for r in reasons))
            cf = counterfactual.suggest(values)
            st.markdown("#### 💡 Как получить одобрение")
            if cf:
                st.caption("Контрфактический анализ: минимальные изменения, при которых "
                           "модель одобрит заявку.")
                for s in cf:
                    st.markdown(
                        f"- **{s['label']}** — {s['action']} → вероятность дефолта снизится "
                        f"до **{s['new_pd']:.1%}** ✅"
                    )
            else:
                st.caption("Одиночных изменений недостаточно — рассмотрите комбинацию мер: "
                           "увеличить первоначальный взнос, снизить сумму кредита и закрыть "
                           "действующие кредиты.")
        else:
            st.success("Заявка соответствует требованиям модели. "
                       "Объяснение факторов доступно выше (право заёмщика на пояснение).")

        # Выгрузка решения в PDF — готовый клиентский документ
        pdf_bytes = report.build_decision_pdf(values)
        if pdf_bytes:
            st.download_button(
                "📄 Скачать решение по заявке (PDF)",
                data=pdf_bytes,
                file_name="reshenie_po_ipoteke.pdf",
                mime="application/pdf",
                width="stretch",
            )

    st.info(
        "🔎 **Связь с ВКР.** Этот раздел реализует **Рекомендацию 3** "
        "(развитие инструментов объяснения решений на базе SHAP) и снимает "
        "риск «чёрного ящика», описанный в разделе 3.2. Решение можно выгрузить "
        "в PDF — как готовый клиентский документ с объяснением и правовой оговоркой."
    )


# =========================================================================== #
# СТРАНИЦА: AVM
# =========================================================================== #
def page_avm():
    hero(
        "🏢 Автоматическая оценка недвижимости (AVM)",
        "Двухступенчатая модель CatBoost → LightGBM, как у сервиса «Домклик» "
        f"(раздел 2.2 ВКР). Средняя погрешность ≈ {avm_meta()['metrics']['mape_final']:.1%}.",
    )

    d = avm.default_property()
    with st.form("avm_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            region = st.selectbox("Регион", C.REGIONS, index=C.REGIONS.index(d["region"]))
            rooms = st.selectbox("Комнат", [1, 2, 3, 4], index=[1, 2, 3, 4].index(d["rooms"]))
            area = st.number_input("Площадь, м²", 25.0, 140.0, float(d["area"]), step=1.0)
            year_built = st.slider("Год постройки", 1960, 2025, int(d["year_built"]))
        with col2:
            floor = st.number_input("Этаж", 1, 40, int(d["floor"]))
            total_floors = st.number_input("Этажей в доме", 2, 40, int(d["total_floors"]))
            wall_material = st.selectbox("Материал стен", C.WALL_MATERIALS,
                                         index=C.WALL_MATERIALS.index(d["wall_material"]))
            renovation = st.selectbox("Состояние ремонта", C.RENOVATION,
                                      index=C.RENOVATION.index(d["renovation"]))
        with col3:
            distance_to_center = st.slider("До центра, км", 0.5, 40.0,
                                           float(d["distance_to_center"]))
            metro_minutes = st.slider("До метро/остановки, мин.", 1, 60, int(d["metro_minutes"]))
            has_balcony = st.checkbox("Балкон/лоджия", value=bool(d["has_balcony"]))
        st.form_submit_button("Оценить объект", type="primary", width="stretch")

    obj = dict(
        area=area, rooms=rooms, floor=floor, total_floors=total_floors,
        year_built=year_built, distance_to_center=distance_to_center,
        metro_minutes=metro_minutes, region=region, wall_material=wall_material,
        renovation=renovation, has_balcony=int(has_balcony),
    )
    res = avm.predict_one(obj)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Оценочная стоимость", fmt_rub(res["price_final"]))
    c2.metric("Цена за м²", fmt_rub(res["price_per_m2"]))
    c3.metric("Доверительный диапазон (±MAPE)",
              f"{fmt_rub(res['low'])} … {fmt_rub(res['high'])}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Этап 1 (CatBoost)", "Этап 2 (LightGBM, финал)"],
        y=[res["price_stage1"], res["price_final"]],
        marker_color=[GREY, GREEN],
        text=[fmt_rub(res["price_stage1"]), fmt_rub(res["price_final"])],
        textposition="outside",
    ))
    fig.update_layout(height=320, yaxis_title="Стоимость, ₽",
                      margin=dict(l=10, r=10, t=30, b=10),
                      title="Двухступенчатая оценка")
    st.plotly_chart(fig, width="stretch")

    st.info(
        "🔎 **Связь с ВКР.** Раздел воспроизводит архитектуру AVM сервиса «Домклик» "
        "(этап 1 — подбор аналогов и базовая оценка, этап 2 — финальная оценка). "
        f"Точность ≈ {res['mape']:.1%} соответствует заявленным в работе 4–5% и "
        "обеспечивает экономию на сторонних оценщиках ≈ 30%."
    )


# =========================================================================== #
# СТРАНИЦА: СПРАВЕДЛИВОСТЬ
# =========================================================================== #
def page_fairness():
    hero(
        "⚖️ Тестирование моделей на справедливость (fairness)",
        "Проверка отсутствия алгоритмической предвзятости по защищённым признакам. "
        "Реализация Рекомендации 2 и принципов Кодекса этики Банка России.",
    )

    attr_map = {"Пол": "gender", "Возрастная группа": "age_group", "Регион": "region"}
    attr_label = st.radio("Защищённый признак", list(attr_map.keys()), horizontal=True,
                          key="fair_attr")
    attr = attr_map[attr_label]

    t = fairness.group_metrics(attr)
    s = fairness.summary(attr)

    c1, c2, c3 = st.columns(3)
    c1.metric("Разрыв одобрений (DP gap)", f"{s['dp_gap']:.1%}")
    c2.metric("Отношение (правило 4/5)", f"{s['dp_ratio']:.2f}",
              "✓ пройдено" if s["passes_4_5"] else "✗ нарушение",
              delta_color="normal" if s["passes_4_5"] else "inverse")
    c3.metric("Разрыв Equal Opportunity", f"{s['eo_gap']:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Доля одобрений (DP)", x=t["Группа"],
                         y=t["Доля одобрений"], marker_color=GREEN))
    fig.add_trace(go.Bar(name="Одобрение «хороших» (EO)", x=t["Группа"],
                         y=t["Одобрение «хороших»"], marker_color="#9FD8B4"))
    fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%",
                      yaxis_title="Доля одобрений",
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    show = t.copy()
    for col in ["Доля одобрений", "Одобрение «хороших»", "Одобрение «плохих»",
                "Факт. дефолтность"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    st.dataframe(show, width="stretch", hide_index=True)

    verdict = ("проходит" if s["passes_4_5"] else "НЕ проходит")
    st.info(
        f"🔎 **Связь с ВКР.** По признаку «{attr_label.lower()}» модель **{verdict}** "
        f"«правило 4/5» (отношение {s['dp_ratio']:.2f}). Наименьшая доля одобрений — "
        f"в группе «{s['worst_group']}». Важно: модель не использует защищённые признаки, "
        "но возможен косвенный эффект через корреляты (доход, регион) — именно это "
        "выявляет fairness-тестирование (раздел 3.2, Рекомендация 2)."
    )


# =========================================================================== #
# СТРАНИЦА: МОНИТОРИНГ
# =========================================================================== #
def page_monitoring():
    hero(
        "📉 Мониторинг моделей (Model Operations Center)",
        "Контроль стабильности модели при изменении макросреды. "
        "Сценарий «шок 2024–2025»: рост ключевой ставки и ухудшение платёжеспособности.",
    )

    r = monitoring.monitoring_report()
    st_text, st_color = r["status_score"]

    c1, c2, c3 = st.columns(3)
    c1.metric("PSI по скоринговому баллу", f"{r['psi_score']:.3f}")
    c2.metric("PSI по ПДН", f"{r['psi_pdn']:.3f}")
    c3.metric("Средний PD: до → после шока",
              f"{r['mean_pd_shock']:.1%}", f"+{(r['mean_pd_shock']-r['mean_pd_base'])*100:.1f} п.п.",
              delta_color="inverse")

    st.markdown(
        f"**Статус модели:** {pill(st_text, {'green': GREEN, 'orange': ORANGE, 'red': RED}[st_color])}",
        unsafe_allow_html=True,
    )
    st.progress(min(r["psi_score"] / 0.5, 1.0))
    st.caption("Пороги PSI: < 0.10 стабильно · 0.10–0.25 наблюдение · > 0.25 ретрейнинг.")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="До шока (обучение)", x=r["bin_labels"],
                         y=r["base_share"], marker_color=GREEN))
    fig.add_trace(go.Bar(name="После шока (текущий период)", x=r["bin_labels"],
                         y=r["shock_share"], marker_color=RED))
    fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%",
                      xaxis_title="Бин скорингового балла (1 — наибольший риск, 10 — наименьший)",
                      yaxis_title="Доля популяции",
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Под действием шока популяция смещается влево (в бины с меньшим баллом / "
        "большим риском) — это и фиксирует PSI."
    )

    a1, a2 = st.columns(2)
    a1.metric("Доля одобрений: до шока", f"{r['approval_base']:.1%}")
    a2.metric("Доля одобрений: после шока", f"{r['approval_shock']:.1%}",
              f"{(r['approval_shock']-r['approval_base'])*100:.1f} п.п.", delta_color="inverse")

    st.warning(
        "🔎 **Связь с ВКР.** Сдвиг распределения (PSI выше порога) — ранний сигнал "
        "деградации модели, как в эпизоде I кв. 2025 г. (рост просрочки на 90% за "
        "квартал, раздел 2.1). Это обосновывает **Рекомендацию 1** (ускоренный "
        "ретрейнинг каждые 3–6 мес.) и **Рекомендацию 4** (центр мониторинга MOC)."
    )


# =========================================================================== #
# СТРАНИЦА: КАРТА РИСКОВ
# =========================================================================== #
def page_risk_map():
    hero(
        "⚠️ Карта рисков применения ИИ в ипотечном кредитовании",
        "Соответствует Таблице 5 и Рисунку 4 ВКР. Оси — вероятность и влияние (1–5).",
    )

    df = risk_map.risk_frame()
    fig = px.scatter(
        df, x="probability", y="impact", size=[28] * len(df), text="Риск",
        color="Риск", hover_data={"Описание": True, "Рекомендация": True,
                                  "probability": False, "impact": False},
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#fff")))
    fig.add_hline(y=3.0, line_dash="dot", line_color=GREY)
    fig.add_vline(x=3.0, line_dash="dot", line_color=GREY)
    fig.add_annotation(x=4.6, y=4.85, text="Критическая зона", showarrow=False,
                       font=dict(color=RED, size=12))
    fig.update_layout(height=480, xaxis=dict(title="Вероятность", range=[2.5, 5]),
                      yaxis=dict(title="Влияние на банк", range=[2.5, 5]),
                      showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Риски и рекомендации по их снижению")
    st.dataframe(df[["Риск", "Описание", "Рекомендация"]], width="stretch",
                 hide_index=True)


# =========================================================================== #
# СТРАНИЦА: ЭФФЕКТЫ
# =========================================================================== #
def page_effects():
    hero(
        "📈 Эффекты внедрения ИИ в ипотечное кредитование",
        "Сопоставление «до / после» по данным раздела 3.1 ВКР (Таблица 4, Рисунок 3).",
    )

    f = C.THESIS_FACTS
    bars = [
        ("Автоодобрения, %", f["auto_approval_before"], f["auto_approval_after"]),
        ("Gini скоринга, %", (f["gini_before_low"] + f["gini_before_high"]) / 2,
         (f["gini_after_low"] + f["gini_after_high"]) / 2),
        ("Просрочка 90+, %", (f["npl90_before_low"] + f["npl90_before_high"]) / 2,
         (f["npl90_after_low"] + f["npl90_after_high"]) / 2),
        ("Расходы на оценку, %", 100, 100 - f["valuer_savings_pct"]),
    ]
    labels = [b[0] for b in bars]
    before = [b[1] for b in bars]
    after = [b[2] for b in bars]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="До ИИ", x=labels, y=before, marker_color=GREY,
                         text=before, textposition="outside"))
    fig.add_trace(go.Bar(name="После ИИ", x=labels, y=after, marker_color=GREEN,
                         text=after, textposition="outside"))
    fig.update_layout(barmode="group", height=420, legend=dict(orientation="h", y=1.1),
                      yaxis_title="Значение показателя",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        | Показатель | До ИИ | После ИИ | Эффект |
        |---|---|---|---|
        | Автоодобрения ипотеки в «Домклик» | 80% | 88% | +8 п.п. |
        | Срок решения по заявке | часы/дни | минуты | в 50–100 раз быстрее |
        | Расходы на сторонних оценщиков | 100% | ≈70% | −30% |
        | Gini скоринговых моделей | 50–60% | 75–80% | +20–25 п.п. |
        | Финансовый эффект ИИ (группа) | — | ≈450 млрд ₽/год | ≈450 млрд ₽/год |
        | Просрочка 90+ дней | 1,3–1,7% | 0,6–0,8% | снижение в 2 раза |
        """
    )
    st.caption("Источник: составлено по данным ВКР (Таблица 4).")


# =========================================================================== #
# СТРАНИЦА: ИИ-КОНСУЛЬТАНТ
# =========================================================================== #
def _llm_cfg():
    """Конфигурация LLM из Streamlit Secrets (GigaChat или OpenAI-совместимый API)."""
    try:
        s = st.secrets
        if "GIGACHAT_CREDENTIALS" in s:
            return {
                "provider": "gigachat",
                "credentials": s["GIGACHAT_CREDENTIALS"],
                "scope": s.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                "model": s.get("LLM_MODEL", "GigaChat"),
            }
        if "LLM_API_KEY" in s:
            return {
                "provider": "openai",
                "api_key": s["LLM_API_KEY"],
                "base_url": s.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                "model": s.get("LLM_MODEL", "gpt-4o-mini"),
            }
    except Exception:
        pass
    return None


def _payment_chart(r: dict):
    yb = r["yearly_balance"]
    fig = go.Figure(go.Scatter(x=list(range(len(yb))), y=yb, fill="tozeroy",
                               line=dict(color=GREEN)))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Год", yaxis_title="Остаток долга, ₽")
    st.plotly_chart(fig, width="stretch")


def page_assistant():
    llm_cfg = _llm_cfg()
    hero(
        "🤖 ИИ-консультант по ипотеке",
        "Клиентский ИИ-интерфейс — 4-й технологический узел из раздела 2.2 ВКР "
        "(концепт GigaChat в «Домклик»).",
    )
    if llm_cfg:
        st.markdown(
            pill("🟢 Подключена языковая модель (LLM)", GREEN)
            + f" &nbsp; <span style='color:#5C6B62'>модель: {llm_cfg.get('model', '—')}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(pill("⚪ Встроенный режим (без ключа)", GREY)
                    + " &nbsp; <span style='color:#5C6B62'>для полноценного диалога "
                    "добавьте ключ LLM в настройках</span>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = [{
            "role": "assistant",
            "content": "Здравствуйте! Я — ИИ-консультант по ипотеке. Спросите про "
                       "программы, ПДН, LTV, документы; попросите рассчитать платёж или "
                       "оценить вашу заявку. Напишите «что ты умеешь» — покажу возможности.",
        }]

    quick = [
        "Что такое ПДН?",
        "Какие программы ипотеки?",
        "Рассчитай платёж 5 млн под 17% на 25 лет",
        "Оцени мою заявку",
    ]
    cols = st.columns(len(quick))
    pending = None
    for i, q in enumerate(quick):
        if cols[i].button(q, key=f"quick_{i}", width="stretch"):
            pending = q

    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])

    msg = st.chat_input("Спросите про ипотеку…") or pending
    if msg:
        st.session_state.chat.append({"role": "user", "content": msg})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg)
        res = assistant.answer(msg, history=st.session_state.chat,
                               last_app=st.session_state.get("last_app"), llm_cfg=llm_cfg)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(res["text"])
            if res.get("payment"):
                _payment_chart(res["payment"])
        st.session_state.chat.append({"role": "assistant", "content": res["text"]})

    st.divider()
    with st.expander("🧮 Калькулятор ипотечного платежа"):
        c1, c2, c3 = st.columns(3)
        principal = c1.number_input("Сумма кредита, ₽", 300_000, 60_000_000,
                                    5_000_000, step=100_000, key="calc_p")
        rate = c2.number_input("Ставка, % годовых", 1.0, 40.0, 17.0, step=0.5, key="calc_r")
        years = c3.number_input("Срок, лет", 1, 30, 20, key="calc_y")
        r = assistant.mortgage_payment(principal, rate, int(years))
        m1, m2, m3 = st.columns(3)
        m1.metric("Ежемесячный платёж", fmt_rub(r["payment"]))
        m2.metric("Переплата", fmt_rub(r["overpay"]), f"{r['overpay_pct']:.0%}",
                  delta_color="inverse")
        m3.metric("Всего выплат", fmt_rub(r["total"]))
        _payment_chart(r)

    st.info(
        "🔎 **Связь с ВКР.** Консультант демонстрирует клиентский ИИ-узел ипотечного "
        "конвейера (концепт GigaChat). Он интегрирован с реальными моделями прототипа: "
        "может оценить заявку (скоринг), объяснить отказ (SHAP) и рассчитать платёж. "
        "При добавлении ключа LLM в настройках превращается в полноценный диалог."
    )


# =========================================================================== #
# СТРАНИЦА: КАРТА НЕДВИЖИМОСТИ
# =========================================================================== #
def page_map():
    hero("🗺️ Карта недвижимости России",
         "Уровень цен по городам и AVM-оценка одной и той же квартиры по стране.")
    df = geo.cities_frame()

    fig = go.Figure(go.Scattergeo(
        lon=df["lon"], lat=df["lat"], text=df["Город"],
        marker=dict(size=df["Объём"] / 2.4 + 7, color=df["Цена_м2"], colorscale="YlGn",
                    showscale=True, colorbar=dict(title="₽/м²"),
                    line=dict(width=0.6, color="#fff")),
        hovertemplate="<b>%{text}</b><br>%{marker.color:,.0f} ₽/м²<extra></extra>",
    ))
    fig.update_geos(showland=True, landcolor="#EAF0EC", showcountries=True,
                    countrycolor="#CBD5CE", showcoastlines=False,
                    bgcolor="rgba(0,0,0,0)", fitbounds="locations", resolution=50)
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("AVM-оценка квартиры по городам")
    c1, c2, c3 = st.columns(3)
    area = c1.slider("Площадь, м²", 25, 120, 55)
    rooms = c2.selectbox("Комнат", [1, 2, 3, 4], index=1)
    material = c3.selectbox("Материал стен", C.WALL_MATERIALS, index=2)

    rows = []
    for _, r in df.iterrows():
        obj = dict(avm.default_property(), area=float(area), rooms=int(rooms),
                   region=r["Регион"], wall_material=material)
        base_price = avm.predict_one(obj)["price_final"]
        factor = r["Цена_м2"] / C.REGION_PRICE_PER_M2[r["Регион"]]
        rows.append((r["Город"], base_price * factor))
    bar = pd.DataFrame(rows, columns=["Город", "Оценка"]).sort_values("Оценка", ascending=False)
    figb = go.Figure(go.Bar(x=bar["Город"], y=bar["Оценка"], marker_color=GREEN,
                            text=[f"{p/1e6:.1f}" for p in bar["Оценка"]], textposition="outside"))
    figb.update_layout(height=360, yaxis_title="AVM-оценка, ₽",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figb, width="stretch")
    st.info("🔎 **Связь с ВКР.** Геоструктура рынка и AVM-оценка залога по регионам "
            "(узел оценки недвижимости, раздел 2.2). Цифры на столбцах — млн ₽.")


# =========================================================================== #
# СТРАНИЦА: АНАЛИТИКА ПОРТФЕЛЯ
# =========================================================================== #
def page_portfolio():
    hero("📊 Аналитика ипотечного портфеля",
         "Структура портфеля и кривая дефолтов — узел прогноза поведения портфеля "
         "(раздел 2.1 ВКР).")
    b = portfolio.booked_loans()
    s = portfolio.summary(b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Кредитов в портфеле", f"{s['n']:,}".replace(",", " "))
    c2.metric("Объём портфеля", f"{s['ead_total']/1e9:.1f} млрд ₽")
    c3.metric("Средний PD", f"{s['mean_pd']:.2%}")
    c4.metric("Ожидаемые потери (EL)", f"{s['el_total']/1e9:.2f} млрд ₽", f"{s['el_pct']:.2%}",
              delta_color="off")

    st.subheader("Структура портфеля")
    pick = st.radio("Разрез", ["Программа", "Регион", "Возраст"], horizontal=True)
    col = {"Программа": "program", "Регион": "region", "Возраст": "age_group"}[pick]
    t = portfolio.structure(col)
    fig = px.pie(t, names="Категория", values="Объём", hole=0.5,
                 color_discrete_sequence=px.colors.sequential.Greens_r)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Кривая дефолтов («горб» на 24–36 мес.)")
    v = portfolio.vintage_default_curve()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=v["Месяц"], y=v["Маржинальная PD"], name="Маржинальная PD (мес.)",
                          marker_color="#9FD8B4"))
    fig2.add_trace(go.Scatter(x=v["Месяц"], y=v["Накопленная дефолтность"], name="Накопленная",
                              yaxis="y2", line=dict(color=GREEN, width=3)))
    peak = int(v.loc[v["Маржинальная PD"].idxmax(), "Месяц"])
    fig2.add_vline(x=peak, line_dash="dot", line_color=RED)
    fig2.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Месяцев с момента выдачи",
        yaxis=dict(title="Маржинальная PD", tickformat=".1%"),
        yaxis2=dict(title="Накопленная дефолтность", overlaying="y", side="right",
                    tickformat=".0%"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig2, width="stretch")
    st.info(f"🔎 **Связь с ВКР.** Вероятность дефолта по ипотеке достигает максимума на "
            f"**{peak}-м месяце** — «горб дефолтов» из раздела 2.1. Точность прогноза "
            "портфеля напрямую влияет на резервы (МСФО 9) и финансовый результат банка.")


# =========================================================================== #
# СТРАНИЦА: СТРЕСС-ТЕСТ ПОРТФЕЛЯ
# =========================================================================== #
@st.cache_data(show_spinner=False)
def _stress(rate: float, income: float, price: float) -> dict:
    return portfolio.stress_test(rate, income, price)


def page_stress():
    hero("🌪️ Стресс-тест ипотечного портфеля",
         "Влияние макрошока на ожидаемые потери портфеля: EL = PD × LGD × EAD (МСФО 9).")
    st.caption("Двигайте ползунки сценария — потери пересчитываются моделью онлайн. "
               "Примеры: умеренный шок (+3 пп / −10% / −15%), кризис (+8 пп / −20% / −30%).")

    c1, c2, c3 = st.columns(3)
    rate = c1.slider("Рост ключевой ставки, п.п.", 0.0, 15.0, 5.0, 0.5)
    income = c2.slider("Падение реальных доходов, %", 0, 30, 10)
    price = c3.slider("Падение цен на жильё, %", 0, 40, 15)

    r = _stress(rate, income, price)
    base, stress = r["base"], r["stress"]

    el_add_txt = f"Дополнительные ожидаемые потери: +{r['el_add']/1e9:.2f} млрд ₽"
    st.markdown(f"#### {pill(el_add_txt, RED)}", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Средний PD", f"{stress['mean_pd']:.2%}",
              f"+{r['pd_delta']*100:.2f} п.п.", delta_color="inverse")
    m2.metric("Ожидаемые потери (EL)", f"{stress['el_total']/1e9:.2f} млрд ₽",
              f"было {base['el_total']/1e9:.2f}", delta_color="off")
    m3.metric("EL к портфелю", f"{stress['el_pct']:.2%}",
              f"+{(stress['el_pct']-base['el_pct'])*100:.2f} п.п.", delta_color="inverse")
    m4.metric("Средний LGD", f"{stress['lgd_mean']:.1%}",
              f"+{(stress['lgd_mean']-base['lgd_mean'])*100:.1f} п.п.", delta_color="inverse")

    labels = ["Средний PD, %", "EL к портфелю, %", "Доля риска (PD>порог), %"]
    base_v = [base["mean_pd"]*100, base["el_pct"]*100, base["npl_share"]*100]
    stress_v = [stress["mean_pd"]*100, stress["el_pct"]*100, stress["npl_share"]*100]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Базовый сценарий", x=labels, y=base_v, marker_color=GREY,
                         text=[f"{x:.2f}" for x in base_v], textposition="outside"))
    fig.add_trace(go.Bar(name="Стресс-сценарий", x=labels, y=stress_v, marker_color=RED,
                         text=[f"{x:.2f}" for x in stress_v], textposition="outside"))
    fig.update_layout(barmode="group", height=380, legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.info("🔎 **Связь с ВКР.** Стресс-тестирование портфеля — ключевой инструмент "
            "управления модельным и кредитным риском (раздел 3.2, Рекомендации 1 и 4). "
            "Падение цен на жильё повышает LGD (хуже обеспечение), рост ставки и падение "
            "доходов повышают PD — потери растут нелинейно.")


# --------------------------------------------------------------------------- #
# Роутинг
# --------------------------------------------------------------------------- #
ROUTES = {
    PAGES[0]: page_overview,
    PAGES[1]: page_scoring,
    PAGES[2]: page_assistant,
    PAGES[3]: page_avm,
    PAGES[4]: page_map,
    PAGES[5]: page_portfolio,
    PAGES[6]: page_stress,
    PAGES[7]: page_fairness,
    PAGES[8]: page_monitoring,
    PAGES[9]: page_risk_map,
    PAGES[10]: page_effects,
}
ROUTES[page]()
