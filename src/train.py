"""
Обучение моделей прототипа «мини-Домклик».

1) Модель кредитного скоринга — градиентный бустинг LightGBM
   (ансамблевый метод, как описано в разделе 2.2 ВКР). Защищённые признаки
   (пол, возрастная группа) В МОДЕЛЬ НЕ ПОДАЮТСЯ и используются только
   для последующей проверки справедливости (fairness).

2) Модель автоматической оценки недвижимости (AVM) — двухступенчатая
   архитектура CatBoost → LightGBM (как у сервиса «Домклик», раздел 2.2 ВКР):
     - этап 1 (CatBoost): базовая оценка стоимости по характеристикам объекта;
     - этап 2 (LightGBM): финальная оценка с учётом прогноза первого этапа.
   Для исключения утечки данных прогнозы этапа 1 на обучающей выборке
   рассчитываются методом out-of-fold (перекрёстная проверка).

Результаты обучения (метрики, перечни признаков, уровни категорий) сохраняются
в каталог models/ и используются интерфейсом приложения.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from . import config as C


# --------------------------------------------------------------------------- #
# Общие утилиты
# --------------------------------------------------------------------------- #
def _as_category(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def load_borrowers_split(seed: int = C.RANDOM_SEED):
    """Детерминированное разбиение заёмщиков на train/test (одно для всего приложения)."""
    df = pd.read_csv(C.BORROWERS_CSV)
    train, test = train_test_split(
        df, test_size=0.25, random_state=seed, stratify=df["default"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. Кредитный скоринг
# --------------------------------------------------------------------------- #
SCORING_CAT_COLS = ["employment_type", "program", "region"]


def train_scoring() -> dict:
    train, test = load_borrowers_split()

    feats = C.SCORING_FEATURES
    X_train = _as_category(train[feats], SCORING_CAT_COLS)
    X_test = _as_category(test[feats], SCORING_CAT_COLS)
    y_train, y_test = train["default"], test["default"]

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=C.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=SCORING_CAT_COLS)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    gini = 2 * auc - 1

    # KS-статистика
    order = np.argsort(proba)
    y_sorted = y_test.to_numpy()[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    ks = float(np.max(np.abs(cum_bad - cum_good)))

    thr = C.PD_APPROVE_THRESHOLD
    approval_rate = float((proba < thr).mean())
    # Доля «плохих» среди одобренных (качество отбора)
    approved = proba < thr
    bad_rate_approved = float(y_test[approved].mean()) if approved.sum() else 0.0
    bad_rate_all = float(y_test.mean())

    # Уровни категорий — чтобы при инференсе коды категорий совпадали
    cat_levels = {col: sorted(train[col].unique().tolist()) for col in SCORING_CAT_COLS}

    meta = {
        "features": feats,
        "numeric": C.SCORING_NUMERIC,
        "categorical": SCORING_CAT_COLS,
        "cat_levels": cat_levels,
        "threshold": thr,
        "metrics": {
            "auc": round(float(auc), 4),
            "gini": round(float(gini), 4),
            "ks": round(ks, 4),
            "approval_rate": round(approval_rate, 4),
            "bad_rate_all": round(bad_rate_all, 4),
            "bad_rate_approved": round(bad_rate_approved, 4),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    }

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, C.SCORING_MODEL_PATH)
    C.SCORING_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[Скоринг] AUC={auc:.3f}  Gini={gini:.3f}  KS={ks:.3f}  "
        f"одобрено={approval_rate:.1%}  "
        f"плохих среди одобренных={bad_rate_approved:.2%} (в среднем {bad_rate_all:.2%})"
    )
    return meta


# --------------------------------------------------------------------------- #
# 2. AVM (двухступенчатая модель оценки недвижимости)
# --------------------------------------------------------------------------- #
AVM_CAT_COLS = ["region", "wall_material", "renovation"]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def train_avm() -> dict:
    df = pd.read_csv(C.PROPERTIES_CSV)
    train, test = train_test_split(df, test_size=0.25, random_state=C.RANDOM_SEED)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    feats = C.AVM_FEATURES
    y_train = np.log(train["price"].to_numpy())
    y_test_price = test["price"].to_numpy()

    cat_idx = [feats.index(c) for c in AVM_CAT_COLS]

    def make_catboost() -> CatBoostRegressor:
        return CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=C.RANDOM_SEED,
            verbose=False,
        )

    # --- Этап 1: out-of-fold прогнозы CatBoost на обучающей выборке ---
    oof_stage1 = np.zeros(len(train))
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    for tr_idx, val_idx in kf.split(train):
        cb = make_catboost()
        cb.fit(
            Pool(train.iloc[tr_idx][feats], y_train[tr_idx], cat_features=cat_idx)
        )
        oof_stage1[val_idx] = cb.predict(
            Pool(train.iloc[val_idx][feats], cat_features=cat_idx)
        )

    # Финальная модель этапа 1 — на всей обучающей выборке (для инференса)
    stage1 = make_catboost()
    stage1.fit(Pool(train[feats], y_train, cat_features=cat_idx))
    test_stage1 = stage1.predict(Pool(test[feats], cat_features=cat_idx))

    # --- Этап 2: LightGBM поверх признаков + прогноз этапа 1 ---
    X2_train = _as_category(train[feats].copy(), AVM_CAT_COLS)
    X2_train["stage1_pred"] = oof_stage1
    X2_test = _as_category(test[feats].copy(), AVM_CAT_COLS)
    X2_test["stage1_pred"] = test_stage1

    stage2 = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=48,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=C.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    stage2.fit(X2_train, y_train, categorical_feature=AVM_CAT_COLS)

    # --- Оценка качества ---
    pred_stage1_price = np.exp(test_stage1)
    pred_final_price = np.exp(stage2.predict(X2_test))

    mape1 = _mape(y_test_price, pred_stage1_price)
    mape2 = _mape(y_test_price, pred_final_price)
    mae2 = float(np.mean(np.abs(y_test_price - pred_final_price)))
    within5 = float(np.mean(np.abs(pred_final_price - y_test_price) / y_test_price <= 0.05))
    ss_res = np.sum((y_test_price - pred_final_price) ** 2)
    ss_tot = np.sum((y_test_price - y_test_price.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    cat_levels = {col: sorted(train[col].unique().tolist()) for col in AVM_CAT_COLS}
    meta = {
        "features": feats,
        "numeric": C.AVM_NUMERIC,
        "categorical": AVM_CAT_COLS,
        "cat_idx": cat_idx,
        "cat_levels": cat_levels,
        "metrics": {
            "mape_stage1": round(mape1, 4),
            "mape_final": round(mape2, 4),
            "mae_final": round(mae2, 0),
            "within_5pct": round(within5, 4),
            "r2": round(r2, 4),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    }

    stage1.save_model(str(C.AVM_STAGE1_PATH))
    joblib.dump(stage2, C.AVM_STAGE2_PATH)
    C.AVM_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[AVM] MAPE этап1={mape1:.2%}  MAPE финал={mape2:.2%}  "
        f"R²={r2:.3f}  в пределах ±5%: {within5:.1%}"
    )
    return meta


def main() -> None:
    print("Обучение моделей прототипа «мини-Домклик»...")
    train_scoring()
    train_avm()
    print("Готово. Артефакты сохранены в каталог models/.")


if __name__ == "__main__":
    main()
