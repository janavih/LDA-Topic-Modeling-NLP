@echo off
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your Python installation and internet connection.
  pause
  exit /b 1
)
python main.py
pause
