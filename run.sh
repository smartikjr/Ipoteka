#!/usr/bin/env bash
# Запуск прототипа «мини-Домклик» одной командой (Linux/macOS).
#   chmod +x run.sh && ./run.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Создаю виртуальное окружение..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Устанавливаю зависимости..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Запускаю приложение (первый запуск обучит модели ~1–2 мин)..."
streamlit run app.py
