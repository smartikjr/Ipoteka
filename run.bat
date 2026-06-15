@echo off
REM Запуск прототипа «мини-Домклик» одной командой (Windows).
cd /d "%~dp0"

if not exist .venv (
  echo Создаю виртуальное окружение...
  python -m venv .venv
)
call .venv\Scripts\activate

echo Устанавливаю зависимости...
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo Запускаю приложение (первый запуск обучит модели ~1-2 мин)...
streamlit run app.py
