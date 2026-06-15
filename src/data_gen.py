"""
Генерация СИНТЕТИЧЕСКИХ данных для прототипа «мини-Домклик».

ВАЖНО: данные не являются реальными данными ПАО Сбербанк или иного банка.
Это искусственно сгенерированный набор, статистические свойства которого
калиброваны под параметры, описанные в ВКР (доля дефолтов, качество скоринга
Gini ≈ 0.75–0.80, погрешность автоматической оценки недвижимости ≈ 4–5%).
Назначение данных — демонстрация МЕТОДОЛОГИИ применения ИИ, а не воспроизведение
конкретных клиентских данных (что невозможно и недопустимо).

Создаются два набора:
  data/borrowers.csv   — заявки на ипотеку с целевым признаком default (дефолт);
  data/properties.csv  — объекты недвижимости с целевым признаком price (цена).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _age_group(age: int) -> str:
    if age < 30:
        return "<30"
    if age < 45:
        return "30–45"
    if age < 55:
        return "45–55"
    return "55+"


# Параметры, управляющие «разделимостью» классов (а значит — итоговым Gini).
# Больше масштаб сигнала и меньше шум -> выше Gini.
# Подобраны так, чтобы Gini модели был ≈ 0.76–0.80 (как в разделе 2.2 ВКР).
DEFAULT_NOISE_STD = 0.45
DEFAULT_SIGNAL_SCALE = 1.40
DEFAULT_BASE = -4.30


# --------------------------------------------------------------------------- #
# Набор данных по заёмщикам (кредитный скоринг)
# --------------------------------------------------------------------------- #
def generate_borrowers(
    n: int = 25_000,
    seed: int = C.RANDOM_SEED,
    noise_std: float = DEFAULT_NOISE_STD,
    signal_scale: float = DEFAULT_SIGNAL_SCALE,
    base: float = DEFAULT_BASE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Демография ---
    age = np.clip(rng.normal(38, 9, n), 21, 70).round().astype(int)
    gender = rng.choice(C.GENDERS, size=n, p=[0.52, 0.48])
    region = rng.choice(C.REGIONS, size=n, p=C.REGION_WEIGHTS)

    employment_type = rng.choice(C.EMPLOYMENT_TYPES, size=n, p=C.EMPLOYMENT_WEIGHTS)
    # Зарплатный клиент почти всегда имеет зарплатный счёт в банке
    has_salary_account = np.where(
        employment_type == "Зарплатный клиент",
        rng.random(n) < 0.97,
        rng.random(n) < 0.25,
    ).astype(int)

    program = rng.choice(C.PROGRAMS, size=n, p=C.PROGRAM_WEIGHTS)

    # Стаж на текущем месте (мес.): у ИП/самозанятых короче
    base_tenure = rng.exponential(48, n)
    tenure_factor = np.where(np.isin(employment_type, ["ИП", "Самозанятый"]), 0.6, 1.0)
    employment_months = np.clip(base_tenure * tenure_factor, 1, 360).round().astype(int)

    # --- Доход (руб./мес.) ---
    # База зависит от региона; небольшой гендерный разрыв (для демонстрации fairness),
    # надбавка за тип занятости.
    region_income_factor = pd.Series(region).map(
        {
            "Москва и МО": 1.55,
            "Санкт-Петербург и ЛО": 1.30,
            "Краснодарский край": 1.05,
            "Свердловская область": 1.00,
            "Прочие регионы": 0.90,
        }
    ).to_numpy()
    gender_income_factor = np.where(gender == "Женский", 0.93, 1.0)  # ~7% разрыв
    emp_income_factor = pd.Series(employment_type).map(
        {
            "Зарплатный клиент": 1.05,
            "Наёмный работник": 1.00,
            "ИП": 1.15,
            "Самозанятый": 0.95,
        }
    ).to_numpy()
    income = (
        rng.lognormal(mean=np.log(85_000), sigma=0.45, size=n)
        * region_income_factor
        * gender_income_factor
        * emp_income_factor
    )
    income = np.clip(income, 30_000, 1_500_000).round(-2)

    # --- Кредитная история и долговая нагрузка ---
    num_existing_loans = rng.poisson(0.8, n).clip(0, 6)
    credit_score = np.clip(
        rng.normal(710, 75, n) - 18 * num_existing_loans + rng.normal(0, 20, n),
        300,
        850,
    ).round().astype(int)
    # ПДН (%) растёт с числом кредитов
    pdn = np.clip(
        rng.normal(28, 12, n) + 6 * num_existing_loans, 3, 80
    ).round(1)

    # --- Параметры сделки ---
    # Стоимость недвижимости связана с доходом (доступность жилья)
    property_value = np.clip(
        income * rng.uniform(35, 70, n), 1_500_000, 60_000_000
    ).round(-4)
    down_payment_pct = np.select(
        [program == "Рыночная", program == "Семейная", program == "IT-ипотека"],
        [
            rng.uniform(20, 50, n),
            rng.uniform(20, 40, n),
            rng.uniform(20, 35, n),
        ],
        default=rng.uniform(15, 30, n),
    ).round(1)
    loan_amount = (property_value * (1 - down_payment_pct / 100)).round(-3)
    ltv = (loan_amount / property_value * 100).round(1)
    term_years = rng.choice(
        [10, 15, 20, 25, 30], size=n, p=[0.07, 0.15, 0.33, 0.30, 0.15]
    )

    # --- Латентный риск дефолта (log-odds) ---
    emp_risk = pd.Series(employment_type).map(
        {
            "Зарплатный клиент": -0.25,
            "Наёмный работник": 0.0,
            "ИП": 0.55,
            "Самозанятый": 0.65,
        }
    ).to_numpy()
    program_risk = pd.Series(program).map(
        {
            "Рыночная": 0.25,
            "Семейная": -0.10,
            "IT-ипотека": -0.20,
            "Льготная (иные)": 0.0,
        }
    ).to_numpy()
    # Возрастной риск: повышен у молодых (<25) и пожилых (>60)
    age_risk = (
        0.018 * ((age - 40) / 10.0) ** 2
        + np.where(age < 25, 0.40, 0.0)
        + np.where(age > 60, 0.55, 0.0)
    )

    signal = (
        0.050 * (pdn - 30)
        - 0.013 * (credit_score - 700)
        + 0.026 * (ltv - 75)
        - 0.55 * np.log(income / 85_000)
        + 0.38 * num_existing_loans
        - 0.65 * has_salary_account
        + 1.25 * emp_risk
        + 1.20 * program_risk
        + age_risk
        - 0.004 * (employment_months / 12.0)
    )
    # base — уровень (доля дефолтов ≈ 7%); signal_scale — сила связи признаков с риском;
    # noise_std — неустранимый шум, ограничивающий достижимый Gini.
    z = base + signal_scale * signal + rng.normal(0, noise_std, n)
    p_default = _sigmoid(z)
    default = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "region": region,
            "income": income,
            "pdn": pdn,
            "credit_score": credit_score,
            "num_existing_loans": num_existing_loans,
            "employment_type": employment_type,
            "employment_months": employment_months,
            "has_salary_account": has_salary_account,
            "program": program,
            "property_value": property_value,
            "loan_amount": loan_amount,
            "ltv": ltv,
            "down_payment_pct": down_payment_pct,
            "term_years": term_years,
            "default": default,
        }
    )
    df["age_group"] = df["age"].map(_age_group)
    return df


# --------------------------------------------------------------------------- #
# Набор данных по объектам недвижимости (AVM)
# --------------------------------------------------------------------------- #
PRICE_NOISE_STD = 0.05  # лог-нормальный шум цены -> погрешность модели ≈ 4–5%


def generate_properties(n: int = 18_000, seed: int = C.RANDOM_SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    region = rng.choice(C.REGIONS, size=n, p=C.REGION_WEIGHTS)
    rooms = rng.choice([1, 2, 3, 4], size=n, p=[0.30, 0.38, 0.24, 0.08])
    # Площадь зависит от числа комнат
    base_area = pd.Series(rooms).map({1: 38, 2: 55, 3: 75, 4: 100}).to_numpy()
    area = np.clip(base_area + rng.normal(0, 7, n), 25, 140).round(1)

    total_floors = rng.integers(5, 26, n)
    floor = (rng.uniform(1, 1, n)).astype(int)
    floor = np.array([rng.integers(1, tf + 1) for tf in total_floors])

    year_built = rng.integers(1960, 2026, n)
    wall_material = rng.choice(C.WALL_MATERIALS, size=n, p=[0.45, 0.30, 0.25])
    renovation = rng.choice(C.RENOVATION, size=n, p=[0.20, 0.55, 0.25])
    has_balcony = (rng.random(n) < 0.8).astype(int)
    distance_to_center = np.clip(rng.exponential(8, n), 0.5, 40).round(1)
    metro_minutes = np.clip(rng.exponential(12, n), 1, 60).round().astype(int)

    # --- Цена ---
    ppm2_base = pd.Series(region).map(C.REGION_PRICE_PER_M2).to_numpy().astype(float)
    mat_mult = pd.Series(wall_material).map(
        {"Панель": 0.92, "Кирпич": 1.00, "Монолит": 1.08}
    ).to_numpy()
    ren_mult = pd.Series(renovation).map(
        {"Без отделки": 0.88, "Типовой ремонт": 1.00, "Евроремонт": 1.10}
    ).to_numpy()
    year_mult = np.clip(1 + 0.004 * (year_built - 1990), 0.85, 1.25)
    floor_mult = np.where(floor == 1, 0.97, np.where(floor == total_floors, 0.98, 1.0))
    dist_mult = np.exp(-0.012 * distance_to_center)
    metro_mult = np.clip(1 - 0.004 * np.maximum(metro_minutes - 5, 0), 0.80, 1.0)
    balcony_mult = np.where(has_balcony == 1, 1.015, 1.0)
    # Лёгкая экономия на масштабе: руб/м² слегка ниже для больших квартир
    scale_mult = np.clip(1.05 - 0.0015 * area, 0.90, 1.05)

    ppm2 = (
        ppm2_base
        * mat_mult
        * ren_mult
        * year_mult
        * floor_mult
        * dist_mult
        * metro_mult
        * balcony_mult
        * scale_mult
    )
    noise = rng.lognormal(mean=0.0, sigma=PRICE_NOISE_STD, size=n)
    price = (area * ppm2 * noise).round(-3)

    df = pd.DataFrame(
        {
            "area": area,
            "rooms": rooms,
            "floor": floor,
            "total_floors": total_floors,
            "year_built": year_built,
            "distance_to_center": distance_to_center,
            "metro_minutes": metro_minutes,
            "region": region,
            "wall_material": wall_material,
            "renovation": renovation,
            "has_balcony": has_balcony,
            "price": price,
        }
    )
    return df


def main() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    borrowers = generate_borrowers()
    borrowers.to_csv(C.BORROWERS_CSV, index=False)
    print(f"borrowers.csv: {len(borrowers):,} строк, "
          f"доля дефолтов = {borrowers['default'].mean():.2%}")

    properties = generate_properties()
    properties.to_csv(C.PROPERTIES_CSV, index=False)
    print(f"properties.csv: {len(properties):,} строк, "
          f"средняя цена = {properties['price'].mean():,.0f} руб.")


if __name__ == "__main__":
    main()
