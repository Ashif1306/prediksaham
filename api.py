"""
api.py
======
REST API untuk Prediksi Harga Saham TLKM (Next Trading Day)
menggunakan FastAPI + SimpleRNN yang sudah di-training.

Prinsip utama:
  - Model & scaler hanya di-load SEKALI saat startup (lifespan)
  - TIDAK ADA fit(), model.save(), atau training ulang
  - Semua inference melalui model yang sudah ada

Cara menjalankan:
  uvicorn api:app --reload

Dokumentasi interaktif (Swagger UI):
  http://127.0.0.1:8000/docs

ReDoc:
  http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── ML / DL imports ────────────────────────────────────────────────────────
from tensorflow import keras
import joblib

# ── Utility kalender BEI (sudah ada di project) ────────────────────────────
from calendar_utils import get_next_trading_day


# =============================================================================
# KONFIGURASI (HARUS SAMA PERSIS DENGAN TRAINING)
# =============================================================================

MODEL_PATH   = "models/tlkm_rnn_model.keras"
SCALER_PATH  = "models/tlkm_scaler.pkl"
DATA_PATH    = "data/data_tlkm_harga_saham.csv"

WINDOW_SIZE  = 10          # Sequence length (timesteps)
TARGET_COL   = "Close"     # Kolom target

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RETURN_LAG_1", "RETURN_LAG_2",
    "RSI_SLOPE", "ROLL_STD_RETURN_5D",
    "MA_5", "MA_10",
]


# =============================================================================
# GLOBAL STATE  (diisi saat startup, read-only setelahnya)
# =============================================================================

# Gunakan dict agar bisa di-mutate di dalam fungsi lifespan
app_state: dict = {
    "model":  None,
    "scaler": None,
}


# =============================================================================
# FEATURE ENGINEERING (IDENTIK DENGAN TRAINING — JANGAN DIUBAH)
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan 11 kolom fitur dari raw OHLCV.
    Fungsi ini adalah mirror persis dari tlkm_rnn_main.py::create_features().
    JANGAN modifikasi tanpa menyesuaikan ulang model.
    """
    df = df.copy()

    if "Close" not in df.columns:
        raise ValueError("Kolom 'Close' tidak ditemukan dalam dataset")

    # Intermediate: RSI-14
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    rsi_14   = 100 - (100 / (1 + rs))

    # Intermediate: return harian
    return_1d = df["Close"].pct_change()

    # 6 fitur teknikal final
    df["RETURN_LAG_1"]       = return_1d.shift(1)
    df["RETURN_LAG_2"]       = return_1d.shift(2)
    df["RSI_SLOPE"]          = rsi_14.diff()
    df["ROLL_STD_RETURN_5D"] = return_1d.rolling(window=5, min_periods=5).std()
    df["MA_5"]               = df["Close"].rolling(window=5,  min_periods=5).mean()
    df["MA_10"]              = df["Close"].rolling(window=10, min_periods=10).mean()

    df = df.dropna()

    FINAL_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "RETURN_LAG_1", "RETURN_LAG_2",
        "RSI_SLOPE", "ROLL_STD_RETURN_5D",
        "MA_5", "MA_10",
    ]
    return df[FINAL_COLS]


# =============================================================================
# LOAD DATA HISTORIS + FEATURE ENGINEERING
# =============================================================================

def load_feature_data() -> pd.DataFrame:
    """
    Load CSV historis, lakukan feature engineering,
    dan kembalikan DataFrame siap-inferensi.
    Dipanggil setiap request /predict agar data selalu terbaru.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File data tidak ditemukan: {DATA_PATH}. "
            "Pastikan training sudah dijalankan terlebih dahulu."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])

    # Pastikan DatetimeIndex valid
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy()
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    # Konversi kolom numerik
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Feature engineering — NO FIT, hanya transformasi deterministik
    df_feat = create_features(df)

    if len(df_feat) < WINDOW_SIZE:
        raise ValueError(
            f"Data tidak cukup setelah feature engineering. "
            f"Tersedia {len(df_feat)} baris, dibutuhkan minimal {WINDOW_SIZE}."
        )

    return df_feat


# =============================================================================
# INFERENCE PIPELINE (NO FIT, NO SAVE)
# =============================================================================

def run_inference(model, scaler, df_feat: pd.DataFrame) -> dict:
    """
    Jalankan prediksi next-trading-day dari data fitur terbaru.

    Args:
        model  : Keras model yang sudah di-load (read-only).
        scaler : MinMaxScaler yang sudah di-fit saat training (read-only).
        df_feat: DataFrame dengan FEATURE_COLS, sudah siap.

    Returns:
        dict: Hasil prediksi lengkap.
    """
    # ── Ekstrak window terakhir ────────────────────────────────────────────
    last_window = df_feat[FEATURE_COLS].iloc[-WINDOW_SIZE:].values  # (10, 11)

    # ── Validasi shape ─────────────────────────────────────────────────────
    expected = (WINDOW_SIZE, len(FEATURE_COLS))
    if last_window.shape != expected:
        raise ValueError(
            f"Shape window tidak sesuai: expected {expected}, got {last_window.shape}"
        )

    # ── Normalisasi TANPA re-fit (scaler sudah fitted dari training) ────────
    scaler_n = getattr(scaler, "n_features_in_", len(getattr(scaler, "scale_", [])))
    if scaler_n != len(FEATURE_COLS):
        raise ValueError(
            f"Mismatch fitur: scaler expects {scaler_n}, pipeline provides {len(FEATURE_COLS)}"
        )

    last_window_scaled = scaler.transform(last_window)          # NO FIT!

    # ── Reshape ke (1, WINDOW_SIZE, n_features) ────────────────────────────
    X_pred = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))

    # ── Model inference ────────────────────────────────────────────────────
    y_pred_scaled = float(model.predict(X_pred, verbose=0).flatten()[0])

    # ── Inverse transform untuk kolom Close saja ───────────────────────────
    target_idx      = FEATURE_COLS.index(TARGET_COL)
    predicted_price = float(
        (y_pred_scaled - scaler.min_[target_idx]) / scaler.scale_[target_idx]
    )

    # ── Tanggal & perubahan harga ───────────────────────────────────────────
    last_actual_date  = df_feat.index[-1]
    last_actual_price = float(df_feat[TARGET_COL].iloc[-1])
    prediction_date   = get_next_trading_day(last_actual_date)

    price_change     = predicted_price - last_actual_price
    price_change_pct = (price_change / last_actual_price) * 100

    return {
        "predicted_price"  : round(predicted_price,   2),
        "prediction_date"  : prediction_date.strftime("%Y-%m-%d"),
        "prediction_day"   : prediction_date.strftime("%A"),
        "last_actual_date" : last_actual_date.strftime("%Y-%m-%d"),
        "last_actual_day"  : last_actual_date.strftime("%A"),
        "last_actual_price": round(last_actual_price, 2),
        "price_change"     : round(price_change,      2),
        "price_change_pct" : round(price_change_pct,  4),
        "trend"            : "bullish" if price_change > 0 else ("bearish" if price_change < 0 else "flat"),
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# LIFESPAN — LOAD MODEL & SCALER SEKALI SAAT STARTUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager FastAPI.
    Model & scaler di-load SATU KALI saat server start,
    dan di-release saat server shutdown.
    Tidak ada fit() atau save() di sini maupun di endpoint manapun.
    """
    print("=" * 60)
    print("🚀 TLKM Stock Prediction API — Starting up...")
    print("=" * 60)

    # Validasi keberadaan file sebelum server menerima request
    for path, label in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler")]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"{label} tidak ditemukan: {path}\n"
                "Jalankan training terlebih dahulu: python tlkm_rnn_main.py"
            )

    # Load model (Keras) — read-only, tidak ada .fit() / .save()
    app_state["model"] = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model loaded  : {MODEL_PATH}")

    # Load scaler (joblib) — read-only, tidak ada .fit() / .fit_transform()
    app_state["scaler"] = joblib.load(SCALER_PATH)
    print(f"  ✓ Scaler loaded : {SCALER_PATH}")

    print("  ✓ API siap menerima request")
    print("=" * 60)

    yield  # ← server berjalan di antara yield ini

    # Cleanup saat shutdown
    app_state["model"]  = None
    app_state["scaler"] = None
    print("👋 API shutdown — resources released.")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title       = "TLKM Stock Price Prediction API",
    description = (
        "REST API untuk prediksi harga penutupan saham PT Telkom Indonesia (TLKM.JK) "
        "pada hari perdagangan berikutnya menggunakan model SimpleRNN.\n\n"
        "**Penting:** API hanya melakukan *inference* — tidak ada training ulang, "
        "tidak ada perubahan pada model atau scaler yang sudah tersimpan."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


# =============================================================================
# RESPONSE SCHEMA (Pydantic)
# =============================================================================

class HealthResponse(BaseModel):
    status      : str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    timestamp   : str = Field(..., example="2026-02-17 09:00:00")
    info        : str = Field(..., example="TLKM Stock Prediction API v1.0.0")


class PredictionResponse(BaseModel):
    predicted_price  : float = Field(..., description="Harga prediksi (Rp)", example=3820.50)
    prediction_date  : str   = Field(..., description="Tanggal prediksi (YYYY-MM-DD)", example="2026-02-18")
    prediction_day   : str   = Field(..., description="Nama hari prediksi", example="Wednesday")
    last_actual_date : str   = Field(..., description="Tanggal data terakhir", example="2026-02-17")
    last_actual_day  : str   = Field(..., description="Nama hari data terakhir", example="Tuesday")
    last_actual_price: float = Field(..., description="Harga penutupan terakhir (Rp)", example=3800.00)
    price_change     : float = Field(..., description="Selisih harga prediksi vs terakhir (Rp)", example=20.50)
    price_change_pct : float = Field(..., description="Persentase perubahan harga (%)", example=0.5395)
    trend            : str   = Field(..., description="Arah tren: bullish / bearish / flat", example="bullish")
    timestamp        : str   = Field(..., description="Waktu prediksi dibuat", example="2026-02-17 09:05:12")


class ErrorResponse(BaseModel):
    error  : str = Field(..., example="File data tidak ditemukan")
    detail : str = Field(..., example="Pastikan training sudah dijalankan")


# =============================================================================
# ENDPOINT: GET /  — Health Check
# =============================================================================

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Memeriksa apakah API berjalan dan model sudah ter-load.",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Mengembalikan status server dan informasi apakah model sudah di-load.
    """
    return HealthResponse(
        status       = "ok",
        model_loaded = app_state["model"] is not None,
        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        info         = "TLKM Stock Prediction API v1.0.0",
    )


# =============================================================================
# ENDPOINT: GET /predict  — Next Trading Day Prediction
# =============================================================================

@app.get(
    "/predict",
    response_model=PredictionResponse,
    summary="Prediksi Harga Next Trading Day",
    description=(
        "Menjalankan inferensi model SimpleRNN untuk memprediksi harga penutupan "
        "TLKM pada hari perdagangan berikutnya.\n\n"
        "**Pipeline:**\n"
        "1. Load data historis terbaru dari CSV\n"
        "2. Feature engineering (deterministik, tanpa fit)\n"
        "3. Normalisasi menggunakan scaler dari training\n"
        "4. Inferensi model (read-only)\n"
        "5. Inverse transform → harga asli\n"
        "6. Tentukan tanggal next trading day (skip weekend & libur BEI)\n\n"
        "⚠️ **Tidak ada training ulang. Model tidak dimodifikasi.**"
    ),
    responses={
        200: {"description": "Prediksi berhasil"},
        500: {"model": ErrorResponse, "description": "Error internal server"},
        503: {"model": ErrorResponse, "description": "Model belum ter-load"},
    },
    tags=["Prediction"],
)
def predict_next_day() -> PredictionResponse:
    """
    Endpoint prediksi utama.

    Mengembalikan harga prediksi TLKM untuk next trading day
    beserta metadata tanggal dan perubahan harga.
    """
    # ── Pastikan model & scaler sudah ter-load ─────────────────────────────
    if app_state["model"] is None or app_state["scaler"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error" : "Model belum ter-load",
                "detail": "Server belum siap. Coba beberapa saat lagi.",
            },
        )

    # ── Load & siapkan data ─────────────────────────────────────────────────
    try:
        df_feat = load_feature_data()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error" : "File data tidak ditemukan",
                "detail": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error" : "Data tidak valid",
                "detail": str(exc),
            },
        ) from exc

    # ── Jalankan inferensi ──────────────────────────────────────────────────
    try:
        result = run_inference(
            model   = app_state["model"],
            scaler  = app_state["scaler"],
            df_feat = df_feat,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error" : "Inferensi gagal",
                "detail": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error" : "Kesalahan tidak terduga saat inferensi",
                "detail": str(exc),
            },
        ) from exc

    return PredictionResponse(**result)


# =============================================================================
# ENTRY POINT (untuk debugging langsung via `python api.py`)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)