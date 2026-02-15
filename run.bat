@echo off
REM Jalankan TLKM Stock Prediction (aktifkan venv dulu jika belum)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python tlkm_stock_prediction_rnn.py
pause
