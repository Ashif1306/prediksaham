@echo off
REM Setup TLKM Stock Prediction - Wajib Python 3.11 atau 3.12
REM TensorFlow belum mendukung Python 3.14

echo ============================================================
echo  TLKM Stock Prediction - Setup
echo ============================================================
echo.

REM Cek apakah Python 3.12 tersedia
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.12 tidak ditemukan!
    echo.
    echo Silakan install Python 3.12 dari: https://www.python.org/downloads/
    echo Centang "Add Python to PATH" saat instalasi.
    echo.
    exit /b 1
)

echo Menggunakan Python 3.12...
py -3.12 -m venv venv
call venv\Scripts\activate.bat

echo.
echo Menginstall dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Instalasi gagal!
    exit /b 1
)

echo.
echo ============================================================
echo  Setup berhasil!
echo  Jalankan: .\run.bat untuk memulai prediksi
echo ============================================================
