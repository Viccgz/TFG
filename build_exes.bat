@echo off
REM Build all executables using PyInstaller. Run this from the project root with the venv activated.
REM Activate venv (PowerShell): .\.venv\Scripts\Activate.ps1

py -m pip install --upgrade pip
py -m pip install pyinstaller

nREM Build main_concurrente
py -m PyInstaller --onefile --paths src --add-data "config.json;." --add-data "data;data" --name main_concurrente src\concurrente\main_concurrente.py

nREM Build main_no_concurrente
py -m PyInstaller --onefile --paths src --add-data "config.json;." --add-data "data;data" --name main_no_concurrente src\concurrente\main_no_concurrente.py

nREM Build metrics
py -m PyInstaller --onefile --paths src --add-data "config.json;." --name metrics src\concurrente\metrics.py

nREM Build emotion_mapper
py -m PyInstaller --onefile --paths src --add-data "config.json;." --name emotion_mapper src\concurrente\dataset_preprocessing\emotion_mapper.py

necho Build finished. Check the dist\ directory for executables.
pause
