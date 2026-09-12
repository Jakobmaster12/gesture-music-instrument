@echo off
REM Doppelklick startet das Instrument.
cd /d "%~dp0"

if not exist venv (
  echo == Lege virtuelle Umgebung an ==
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

python tools\check_setup.py
python main.py %*
pause
