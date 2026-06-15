@echo off
chcp 65001 >nul
setlocal
REM Запуск прототипа «мини-Домклик» одной командой (Windows).
cd /d "%~dp0"

REM --- Поиск Python: сначала лаунчер py, затем python ---
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py"
if not defined PYEXE ( where python >nul 2>nul && set "PYEXE=python" )

if not defined PYEXE (
  echo.
  echo [ОШИБКА] Python не найден на этом компьютере.
  echo Установите Python с сайта https://python.org/downloads
  echo и ОБЯЗАТЕЛЬНО отметьте галочку "Add python.exe to PATH" при установке.
  echo После установки запустите run.bat снова.
  echo.
  pause
  exit /b 1
)

if not exist .venv (
  echo Создаю виртуальное окружение...
  %PYEXE% -m venv .venv || ( echo [ОШИБКА] Не удалось создать окружение. & pause & exit /b 1 )
)
call .venv\Scripts\activate

echo Устанавливаю зависимости (первый раз это занимает несколько минут)...
python -m pip install --upgrade pip
pip install -r requirements.txt || ( echo [ОШИБКА] Не удалось установить зависимости. & pause & exit /b 1 )

echo.
echo Запускаю приложение (первый запуск обучит модели ~1-2 мин)...
echo Когда откроется браузер - приложение готово. Это окно не закрывайте.
echo.
streamlit run app.py

echo.
echo Приложение остановлено. Можно закрыть это окно.
pause
