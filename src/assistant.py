"""
ИИ-консультант по ипотеке — клиентский интерфейс на основе ИИ.

Соответствует четвёртому технологическому узлу из раздела 2.2 ВКР
(клиентские интерфейсы на базе ИИ, концепт GigaChat в сервисе «Домклик»).

Реализует два режима:
  - встроенный (без ключа): распознавание намерений + извлечение ответа из базы
    знаний методом TF-IDF (символьные n-граммы устойчивы к словоформам), а также
    вызов реальных моделей скоринга и оценки недвижимости;
  - LLM (опционально): если в Streamlit Secrets заданы ключи доступа к
    OpenAI-совместимому API (в т.ч. GigaChat через шлюз), свободные вопросы
    направляются языковой модели с системным промптом по теме ипотеки.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------- #
# База знаний по ипотеке (вопрос/ключевые слова -> ответ)
# --------------------------------------------------------------------------- #
KNOWLEDGE = [
    {
        "q": "что такое пдн показатель долговой нагрузки",
        "a": "**ПДН (показатель долговой нагрузки)** — отношение ежемесячных платежей "
             "по всем кредитам к доходу заёмщика. Чем ниже ПДН, тем выше шансы на "
             "одобрение. При ПДН выше 50–60% банк закладывает повышенный риск, а ЦБ "
             "применяет надбавки к капиталу.",
    },
    {
        "q": "что такое ltv первоначальный взнос отношение кредита к залогу",
        "a": "**LTV (Loan-to-Value)** — отношение суммы кредита к стоимости залога. "
             "Например, при взносе 30% LTV = 70%. Чем меньше LTV (больше первоначальный "
             "взнос), тем ниже риск для банка и выше вероятность одобрения.",
    },
    {
        "q": "какие ипотечные программы бывают семейная it льготная рыночная",
        "a": "Основные программы: **рыночная** ипотека (ставка близка к ключевой), "
             "**семейная** (господдержка для семей с детьми), **IT-ипотека** (для "
             "сотрудников IT-компаний), **льготные** (Дальневосточная, Арктическая и др.). "
             "По данным работы, около 80% выдач Сбербанка приходится на льготные программы.",
    },
    {
        "q": "какой нужен первоначальный взнос минимальный",
        "a": "Обычно **первоначальный взнос от 15–20%** стоимости жилья. По рыночным "
             "программам банки часто требуют 20–30%. Чем больше взнос — тем ниже LTV, "
             "ставка и риск отказа.",
    },
    {
        "q": "какие документы нужны для ипотеки оформление",
        "a": "Как правило: паспорт, подтверждение дохода (справка/выписка), сведения о "
             "занятости, документы по объекту недвижимости. В цифровых сервисах часть "
             "данных подтягивается автоматически (Госуслуги, Госключ), что ускоряет сделку.",
    },
    {
        "q": "что влияет на одобрение заявки факторы решение",
        "a": "На решение влияют: **доход и ПДН**, **кредитная история**, **первоначальный "
             "взнос (LTV)**, число действующих кредитов, тип занятости и стаж. В прототипе "
             "вклад каждого фактора виден в разделе «Скоринг и объяснение (SHAP)».",
    },
    {
        "q": "как принимается решение скоринг модель искусственный интеллект",
        "a": "Решение принимает **скоринговая модель машинного обучения** (градиентный "
             "бустинг). Она оценивает вероятность дефолта (PD) по сотням признаков за доли "
             "секунды. В прототипе Gini модели ≈ 0,78, как и заявлено в работе (0,75–0,80).",
    },
    {
        "q": "почему отказали в ипотеке причина отказа",
        "a": "Чаще всего отказ связан с **высоким ПДН, низким баллом кредитной истории, "
             "малым первоначальным взносом или большим числом кредитов**. В прототипе "
             "конкретные причины показывает SHAP, а раздел «Скоринг» предлагает, что "
             "изменить для одобрения. Спросите: «почему отказ» — я разберу вашу заявку.",
    },
    {
        "q": "что такое avm автоматическая оценка недвижимости стоимость квартиры",
        "a": "**AVM (Automated Valuation Model)** — модель автоматической оценки "
             "недвижимости. В «Домклик» это двухступенчатая модель CatBoost → LightGBM "
             "с погрешностью ≈ 4–5%. Спросите «оцени квартиру» или откройте раздел «AVM».",
    },
    {
        "q": "можно ли досрочно погасить ипотеку",
        "a": "Да, ипотеку можно гасить досрочно — частично или полностью, без штрафов "
             "(по закону). Досрочное погашение снижает переплату по процентам. Эффект "
             "можно прикинуть в калькуляторе платежа ниже.",
    },
    {
        "q": "что такое аннуитетный платёж как рассчитать ежемесячный",
        "a": "**Аннуитетный платёж** — равный ежемесячный платёж на весь срок кредита. "
             "Считается по сумме кредита, ставке и сроку. Напишите, например: "
             "«рассчитай платёж 6 млн под 18% на 20 лет» — я посчитаю.",
    },
    {
        "q": "что такое ключевая ставка как влияет на ипотеку",
        "a": "**Ключевая ставка** Банка России определяет стоимость денег в экономике. "
             "Её рост удорожает рыночную ипотеку и повышает значимость льготных программ. "
             "Резкий рост ставки в 2024 г. — пример макрошока (см. раздел «Мониторинг»).",
    },
    {
        "q": "что такое shap объяснение решения интерпретация",
        "a": "**SHAP** — метод, который раскладывает решение модели на вклад каждого "
             "фактора. Это снимает проблему «чёрного ящика» и выполняет требование "
             "ст. 16 ФЗ-152 об объяснении автоматических решений (Рекомендация 3 работы).",
    },
    {
        "q": "что такое pd вероятность дефолта",
        "a": "**PD (Probability of Default)** — вероятность того, что заёмщик допустит "
             "дефолт. Модель оценивает PD по заявке; если PD ниже порога — заявка "
             "одобряется автоматически.",
    },
    {
        "q": "налоговый вычет по ипотеке возврат налога",
        "a": "При покупке жилья в ипотеку можно получить **имущественный налоговый вычет** "
             "(до 13% от стоимости в пределах лимита) и вычет по уплаченным процентам. "
             "Это снижает фактическую стоимость кредита для заёмщика.",
    },
    {
        "q": "gigachat ии в сбербанке домклик технологии",
        "a": "В работе показано, что Сбербанк применяет ИИ во всём ипотечном конвейере: "
             "AVM-оценка, скоринг, прогноз портфеля и клиентские интерфейсы на базе "
             "**GigaChat** в «Домклик». Этот консультант — демонстрация именно такого "
             "клиентского ИИ-узла, интегрированного с моделями скоринга и оценки.",
    },
]

GREETINGS = ("привет", "здравствуй", "добрый", "хай", "доброе утро", "добрый день")
THANKS = ("спасибо", "благодар", "спс")
HELP_WORDS = ("что ты умеешь", "помощь", "help", "что можешь", "возможности")

CAPABILITIES = (
    "Я — ИИ-консультант по ипотеке. Могу:\n"
    "- ответить на вопросы об ипотеке (ПДН, LTV, программы, документы, ставки);\n"
    "- **рассчитать платёж** — напишите «платёж 6 млн под 18% на 20 лет»;\n"
    "- **оценить заявку** — спросите «одобрят ли» (после расчёта в разделе «Скоринг»);\n"
    "- **объяснить отказ** — спросите «почему отказ»;\n"
    "- подсказать про **оценку недвижимости** (AVM).\n\n"
    "О чём расскажу подробнее?"
)


# --------------------------------------------------------------------------- #
# Извлечение ответа из базы знаний (TF-IDF, символьные n-граммы)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _kb_index():
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vec.fit_transform([item["q"] for item in KNOWLEDGE])
    return vec, matrix


def retrieve(query: str) -> tuple[str | None, float]:
    vec, matrix = _kb_index()
    qv = vec.transform([query.lower()])
    sims = (matrix @ qv.T).toarray().ravel()
    idx = int(np.argmax(sims))
    return KNOWLEDGE[idx]["a"], float(sims[idx])


def retrieve_top(query: str, k: int = 3) -> list[tuple[str, float]]:
    vec, matrix = _kb_index()
    qv = vec.transform([query.lower()])
    sims = (matrix @ qv.T).toarray().ravel()
    order = sims.argsort()[::-1][:k]
    return [(KNOWLEDGE[i]["a"], float(sims[i])) for i in order]


# --------------------------------------------------------------------------- #
# Калькулятор аннуитетного платежа
# --------------------------------------------------------------------------- #
def mortgage_payment(principal: float, annual_rate_pct: float, years: int) -> dict:
    n = int(years * 12)
    mr = annual_rate_pct / 100 / 12
    if mr <= 0:
        payment = principal / n
    else:
        payment = principal * mr * (1 + mr) ** n / ((1 + mr) ** n - 1)
    total = payment * n
    overpay = total - principal

    # Остаток долга по годам (для графика)
    balance = principal
    yearly_balance = [principal]
    for m in range(1, n + 1):
        interest = balance * mr
        balance = balance - (payment - interest)
        if m % 12 == 0:
            yearly_balance.append(max(balance, 0))
    return {
        "payment": payment,
        "total": total,
        "overpay": overpay,
        "overpay_pct": overpay / principal if principal else 0,
        "yearly_balance": yearly_balance,
    }


# --------------------------------------------------------------------------- #
# Разбор запроса на расчёт платежа
# --------------------------------------------------------------------------- #
def _to_number(num_str: str, suffix: str) -> float:
    val = float(num_str.replace(" ", "").replace(",", "."))
    s = suffix.lower()
    if s.startswith(("млн", "миллион")):
        val *= 1_000_000
    elif s.startswith(("тыс", "т.р", "тр", "к", "k")):
        val *= 1_000
    return val


def parse_payment_query(text: str) -> dict | None:
    t = text.lower()
    rate = None
    m = re.search(r"(\d+[.,]?\d*)\s*%", t)
    if m:
        rate = float(m.group(1).replace(",", "."))

    years = None
    m = re.search(r"(\d+)\s*(лет|год|года|г\.)", t)
    if m:
        years = int(m.group(1))

    principal = None
    m = re.search(r"(\d[\d\s.,]*)\s*(млн|миллион\w*|тыс\w*|т\.р|тр|к|k)?", t)
    if m and m.group(1).strip():
        cand = _to_number(m.group(1), m.group(2) or "")
        if cand >= 100_000:           # отсекаем числа ставки/срока
            principal = cand
    if principal and rate and years:
        return {"principal": principal, "rate": rate, "years": years}
    return None


# --------------------------------------------------------------------------- #
# LLM-режим (опционально, OpenAI-совместимый API)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Ты — ИИ-консультант по ипотечному кредитованию в российском банке, часть "
    "клиентского интерфейса ипотечной ИИ-платформы (как «Домклик»). Помогай вежливо "
    "и по делу. Отвечай по-русски, кратко (2–5 предложений), простым языком, при "
    "необходимости списком; используй markdown. Темы: ипотечные программы (рыночная, "
    "семейная, IT, льготные), ставки, ПДН, LTV, первоначальный взнос, документы, "
    "скоринг, оценка недвижимости (AVM), досрочное погашение, налоговый вычет. "
    "Если есть «Справочная информация» — опирайся на неё. Не выдумывай точные ставки "
    "и условия конкретного банка; при неуверенности предложи уточнить у банка. "
    "Если вопрос не про ипотеку — мягко верни разговор к теме."
)


def _llm_context(query: str, last_app: dict | None) -> str:
    """Контекст для LLM: релевантные фрагменты базы знаний + статус заявки."""
    parts = []
    snippets = [a for a, s in retrieve_top(query, 3) if s > 0.05]
    if snippets:
        parts.append("Справочная информация по теме:\n" + "\n".join(f"- {s}" for s in snippets))
    if last_app:
        from . import scoring

        res = scoring.predict_one(last_app)
        verdict = "одобрено" if res["approved"] else "отказано"
        parts.append(
            f"Контекст пользователя: есть рассчитанная заявка — решение «{verdict}», "
            f"вероятность дефолта {res['pd']:.1%}, балл {res['score']}/1000."
        )
    return "\n\n".join(parts)


_GC_TOKEN: tuple[str, float] | None = None


def _build_messages(query: str, history: list[dict], context: str) -> list[dict]:
    system = SYSTEM_PROMPT + (f"\n\n{context}" if context else "")
    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:
        if h["role"] in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": query})
    return messages


def _openai_chat(messages: list[dict], cfg: dict) -> str:
    """OpenAI-совместимый API (OpenAI, Groq и т.п.)."""
    import requests

    resp = requests.post(
        f"{cfg['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json={"model": cfg.get("model", "gpt-4o-mini"), "messages": messages,
              "temperature": 0.3, "max_tokens": 500},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _gigachat_token(cfg: dict) -> str:
    """OAuth-токен GigaChat (кэшируется ~30 минут)."""
    global _GC_TOKEN
    import time
    import uuid

    import requests
    import urllib3

    urllib3.disable_warnings()
    now = time.time()
    if _GC_TOKEN and _GC_TOKEN[1] - 60 > now:
        return _GC_TOKEN[0]
    resp = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={"Authorization": f"Basic {cfg['credentials']}",
                 "RqUID": str(uuid.uuid4()),
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"scope": cfg.get("scope", "GIGACHAT_API_PERS")},
        timeout=20, verify=False,  # GigaChat использует российский корневой сертификат
    )
    resp.raise_for_status()
    j = resp.json()
    exp = j.get("expires_at", int((now + 1800) * 1000)) / 1000
    _GC_TOKEN = (j["access_token"], exp)
    return _GC_TOKEN[0]


def _gigachat_chat(messages: list[dict], cfg: dict) -> str:
    """API GigaChat (Сбербанк)."""
    import requests
    import urllib3

    urllib3.disable_warnings()
    token = _gigachat_token(cfg)
    resp = requests.post(
        "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": cfg.get("model", "GigaChat"), "messages": messages, "temperature": 0.3},
        timeout=30, verify=False,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def llm_answer(query: str, history: list[dict], cfg: dict, context: str = "") -> str | None:
    """Ответ языковой модели. Поддерживает OpenAI-совместимые API и GigaChat.

    При любой ошибке возвращает None (вызывающий код переходит во встроенный режим).
    """
    try:
        messages = _build_messages(query, history, context)
        if cfg.get("provider") == "gigachat":
            return _gigachat_chat(messages, cfg)
        return _openai_chat(messages, cfg)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Главная функция ответа
# --------------------------------------------------------------------------- #
RETRIEVE_THRESHOLD = 0.18


def answer(query: str, history: list[dict] | None = None,
           last_app: dict | None = None, llm_cfg: dict | None = None) -> dict:
    """
    Вернуть ответ консультанта.
    Поля результата: text, source ('rule'|'calc'|'model'|'llm'|'kb'), payment (опц.).
    """
    history = history or []
    t = query.lower().strip()

    # 1. Приветствие / благодарность / помощь
    if any(w in t for w in GREETINGS) and len(t) < 30:
        return {"text": "Здравствуйте! " + CAPABILITIES, "source": "rule"}
    if any(w in t for w in THANKS):
        return {"text": "Пожалуйста! Если будут вопросы по ипотеке — спрашивайте.",
                "source": "rule"}
    if any(w in t for w in HELP_WORDS):
        return {"text": CAPABILITIES, "source": "rule"}

    # 2. Намерение: рассчитать платёж
    if any(w in t for w in ("платёж", "платеж", "рассчита", "посчита", "ежемесячн")):
        parsed = parse_payment_query(t)
        if parsed:
            r = mortgage_payment(parsed["principal"], parsed["rate"], parsed["years"])
            text = (
                f"При сумме {parsed['principal']:,.0f} ₽, ставке {parsed['rate']:g}% и сроке "
                f"{parsed['years']} лет:\n\n"
                f"- **Ежемесячный платёж: {r['payment']:,.0f} ₽**\n"
                f"- Общая выплата: {r['total']:,.0f} ₽\n"
                f"- Переплата: {r['overpay']:,.0f} ₽ ({r['overpay_pct']:.0%})"
            ).replace(",", " ")
            return {"text": text, "source": "calc", "payment": r}
        return {"text": "Подскажите три параметра, например: «рассчитай платёж "
                        "**6 млн** под **18%** на **20 лет**». Или используйте калькулятор ниже.",
                "source": "rule"}

    # 3. Намерение: оценить заявку / почему отказ
    if any(w in t for w in ("одобр", "почему отказ", "моя заявка", "оцени заявк",
                            "шанс", "пройду ли", "дадут ли")):
        return {"text": _assess_last_app(last_app), "source": "model"}

    # 4. Намерение: оценка недвижимости
    if any(w in t for w in ("оцени квартир", "сколько стоит", "оценка недвиж",
                            "стоимость квартир", "оцени недвиж")):
        return {"text": "Оценю объект автоматической моделью AVM. Перейдите в раздел "
                        "**«Оценка недвижимости (AVM)»** слева и введите параметры квартиры — "
                        "модель CatBoost → LightGBM вернёт стоимость с погрешностью ≈ 4–5%.",
                "source": "rule"}

    # 5. Свободный вопрос: если подключён LLM — отвечает он (с опорой на базу знаний)
    if llm_cfg:
        llm = llm_answer(query, history, llm_cfg, context=_llm_context(t, last_app))
        if llm:
            return {"text": llm, "source": "llm"}

    # 6. База знаний
    kb_text, score = retrieve(t)
    if score >= RETRIEVE_THRESHOLD:
        return {"text": kb_text, "source": "kb"}

    # 7. Вежливый фолбэк
    return {"text": "Не уверен, что понял вопрос. Я консультирую по ипотеке: "
                    "программы, ставки, ПДН, LTV, первоначальный взнос, документы, "
                    "расчёт платежа, оценка заявки и недвижимости. Попробуйте "
                    "переформулировать или напишите «что ты умеешь».",
            "source": "rule"}


def _assess_last_app(last_app: dict | None) -> str:
    if not last_app:
        return ("Чтобы оценить заявку, сначала заполните параметры в разделе "
                "**«Скоринг и объяснение»** и нажмите «Рассчитать решение». "
                "После этого я разберу результат здесь.")
    from . import explain, scoring

    res = scoring.predict_one(last_app)
    if res["approved"]:
        return (f"По вашей последней заявке модель даёт **одобрение** "
                f"(вероятность дефолта {res['pd']:.1%}, балл {res['score']}/1000). "
                "Ключевые факторы — в разделе «Скоринг и объяснение».")
    reasons = explain.top_reasons(last_app, k=3)
    return (f"По вашей последней заявке модель даёт **отказ** "
            f"(вероятность дефолта {res['pd']:.1%}). Основные причины:\n"
            + "\n".join(f"- {r}" for r in reasons)
            + "\n\nЧто можно изменить для одобрения — смотрите подсказки в разделе «Скоринг».")
