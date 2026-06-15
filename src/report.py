"""
Генерация PDF-решения по ипотечной заявке — готовый клиентский документ.

Формирует одностраничный (при необходимости многостраничный) документ с:
- параметрами заявки;
- решением (одобрено/отказано), вероятностью дефолта и баллом;
- объяснением решения по факторам (SHAP, Рекомендация 3 ВКР);
- для отказов — подсказками «как повысить шансы на одобрение» (контрфактический анализ);
- правовой оговоркой (ст. 16 ФЗ-152 о праве на пересмотр) и пометкой о
  демонстрационном характере данных.

Кириллица поддерживается через шрифты DejaVu (поставляются вместе с matplotlib).
Если библиотека reportlab недоступна, build_decision_pdf вернёт None.
"""

from __future__ import annotations

import datetime
import hashlib
import io
from pathlib import Path

from . import config as C

try:
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAVE_REPORTLAB = True
except Exception:  # pragma: no cover
    HAVE_REPORTLAB = False

GREEN = "#21A038"
RED = "#E23C3C"
DARK = "#15281C"
GREY = "#8A9A90"
LIGHT = "#F2F6F3"

_FONTS_READY = False


def _register_fonts() -> None:
    global _FONTS_READY
    if _FONTS_READY:
        return
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    _FONTS_READY = True


# --------------------------------------------------------------------------- #
# Форматирование значений заявки
# --------------------------------------------------------------------------- #
def _fmt(feature: str, v) -> str:
    if feature in ("income", "loan_amount"):
        return f"{v:,.0f} ₽".replace(",", " ")
    if feature in ("pdn", "ltv", "down_payment_pct"):
        return f"{v:g}%"
    if feature == "has_salary_account":
        return "Да" if v else "Нет"
    if feature == "term_years":
        return f"{v} лет"
    if feature == "employment_months":
        return f"{v} мес."
    return str(v)


# Порядок и состав полей заявки в документе
_FIELDS = [
    "age", "gender", "region", "income", "pdn", "credit_score",
    "num_existing_loans", "employment_type", "employment_months",
    "has_salary_account", "program", "loan_amount", "down_payment_pct",
    "ltv", "term_years",
]
_LABELS = {**C.FEATURE_LABELS_RU, "gender": "Пол"}


def _app_number(values: dict) -> str:
    raw = f"{values}-{datetime.datetime.now():%Y%m%d%H%M%S}"
    return hashlib.md5(raw.encode()).hexdigest()[:8].upper()


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def build_decision_pdf(values: dict) -> bytes | None:
    """Сформировать PDF-решение по заявке. Возвращает байты PDF или None."""
    if not HAVE_REPORTLAB:
        return None
    from . import counterfactual, explain, scoring

    _register_fonts()
    res = scoring.predict_one(values)
    approved = res["approved"]
    accent = GREEN if approved else RED

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="DejaVu",
                          fontSize=9.5, leading=13, textColor=colors.HexColor(DARK))
    h_title = ParagraphStyle("title", parent=styles["Title"], fontName="DejaVu-Bold",
                             fontSize=17, textColor=colors.HexColor(DARK), spaceAfter=2)
    h_sub = ParagraphStyle("sub", parent=body, fontSize=9, textColor=colors.HexColor(GREY))
    h_sec = ParagraphStyle("sec", parent=body, fontName="DejaVu-Bold", fontSize=11,
                           textColor=colors.HexColor(DARK), spaceBefore=10, spaceAfter=4)
    verdict_style = ParagraphStyle("verdict", parent=body, fontName="DejaVu-Bold",
                                   fontSize=15, textColor=colors.white, alignment=TA_CENTER)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10,
                           textColor=colors.HexColor(GREY))

    story = []

    # Шапка
    story.append(Paragraph("мини-Домклик · Решение по ипотечной заявке", h_title))
    story.append(Paragraph(
        f"Автоматизированная оценка · заявка № {_app_number(values)} · "
        f"{datetime.datetime.now():%d.%m.%Y %H:%M}", h_sub))
    story.append(Spacer(1, 8))

    # Блок решения
    pd_txt = f"Вероятность дефолта (PD): {res['pd']:.1%}"
    score_txt = f"Скоринговый балл: {res['score']} / 1000"
    verdict = Table(
        [[Paragraph(("ЗАЯВКА ОДОБРЕНА" if approved else "В ЗАЯВКЕ ОТКАЗАНО"), verdict_style)],
         [Paragraph(f"<font color='white'>{pd_txt}&nbsp;&nbsp;|&nbsp;&nbsp;{score_txt}"
                    f"&nbsp;&nbsp;|&nbsp;&nbsp;порог: PD &lt; {res['threshold']:.0%}</font>",
                    ParagraphStyle("v2", parent=verdict_style, fontName="DejaVu", fontSize=9.5))]],
        colWidths=[170 * mm])
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(accent)),
        ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 0), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(verdict)
    story.append(Spacer(1, 4))

    # Параметры заявки (две колонки пар «label: value»)
    story.append(Paragraph("Параметры заявки", h_sec))
    pairs = [(f"<b>{_LABELS.get(f, f)}:</b> {_fmt(f, values[f])}") for f in _FIELDS if f in values]
    rows, half = [], (len(pairs) + 1) // 2
    for i in range(half):
        left = Paragraph(pairs[i], body)
        right = Paragraph(pairs[i + half], body) if i + half < len(pairs) else Paragraph("", body)
        rows.append([left, right])
    ptable = Table(rows, colWidths=[85 * mm, 85 * mm])
    ptable.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT)),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor(LIGHT), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(ptable)

    # Объяснение решения (SHAP)
    story.append(Paragraph("Объяснение решения (SHAP)", h_sec))
    story.append(Paragraph(
        "Вклад ключевых факторов в оценку риска (метод SHAP). "
        "«Повышает риск» — в сторону отказа, «снижает риск» — в сторону одобрения.", small))
    contribs = explain.explain_one(values)["contributions"][:6]
    srows = [[Paragraph("<b>Фактор</b>", body), Paragraph("<b>Значение</b>", body),
              Paragraph("<b>Влияние на риск</b>", body)]]
    for c in contribs:
        up = c["shap"] > 0
        eff = (f"<font color='{RED if up else GREEN}'>"
               f"{'▲ повышает' if up else '▼ снижает'} ({c['shap']:+.2f})</font>")
        srows.append([Paragraph(c["label"], body),
                      Paragraph(_fmt(c["feature"], c["value"]), body),
                      Paragraph(eff, body)])
    stab = Table(srows, colWidths=[78 * mm, 42 * mm, 50 * mm])
    stab.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(GREY)),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#E0E6E2")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(stab)

    # Подсказки по одобрению (для отказов)
    if not approved:
        sug = counterfactual.suggest(values)
        if sug:
            story.append(Paragraph("Как повысить шансы на одобрение", h_sec))
            for s in sug:
                story.append(Paragraph(
                    f"• <b>{s['label']}</b> — {s['action']} → вероятность дефолта "
                    f"снизится до <font color='{GREEN}'>{s['new_pd']:.1%}</font>", body))

    # Правовая оговорка
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Решение сформировано автоматизированной моделью. В соответствии со ст. 16 "
        "Федерального закона № 152-ФЗ «О персональных данных» заёмщик вправе требовать "
        "пересмотра решения с участием уполномоченного сотрудника банка. "
        "Документ сформирован демонстрационным прототипом «мини-Домклик» на синтетических "
        "данных, не является офертой и официальным решением кредитной организации.", small))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm,
                            title="Решение по ипотечной заявке")
    doc.build(story)
    return buf.getvalue()
