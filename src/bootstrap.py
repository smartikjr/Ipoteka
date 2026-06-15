"""
Автоматическая подготовка данных и моделей при первом запуске.

Если артефакты (данные, обученные модели) отсутствуют — генерирует данные
и обучает модели. Это позволяет приложению работать сразу после клонирования
репозитория, без отдельных ручных шагов.
"""

from __future__ import annotations

from . import config as C

ARTIFACTS = [
    C.BORROWERS_CSV,
    C.PROPERTIES_CSV,
    C.SCORING_MODEL_PATH,
    C.SCORING_META_PATH,
    C.AVM_STAGE1_PATH,
    C.AVM_STAGE2_PATH,
    C.AVM_META_PATH,
]


def artifacts_ready() -> bool:
    """True, если все данные и модели на месте."""
    return all(p.exists() for p in ARTIFACTS)


def ensure_artifacts(log=print) -> bool:
    """Сгенерировать данные и обучить модели, если их нет. Возвращает True, если что-то строили."""
    if artifacts_ready():
        return False
    # Импортируем здесь, чтобы не тянуть тяжёлые зависимости без необходимости
    from . import data_gen, train

    log("Генерация синтетических данных...")
    data_gen.main()
    log("Обучение моделей (скоринг + AVM)...")
    train.main()
    return True


if __name__ == "__main__":
    built = ensure_artifacts()
    print("Готово." if built else "Артефакты уже на месте — ничего делать не нужно.")
