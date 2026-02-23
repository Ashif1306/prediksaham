"""
api.py — TLKM Stock Prediction API (Fixed)
==========================================
Cara menjalankan:
  uvicorn api:app --reload

Akses:
  http://127.0.0.1:8000/        ← Dashboard HTML
  http://127.0.0.1:8000/predict ← JSON prediksi
  http://127.0.0.1:8000/history ← JSON riwayat
  http://127.0.0.1:8000/docs    ← Swagger UI
"""

from __future__ import annotations

import os
import sqlite3
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tensorflow import keras
import joblib
import yfinance as yf

from calendar_utils import get_next_trading_day


# =============================================================================
# KONFIGURASI
# =============================================================================
MODEL_PATH    = "models/tlkm_rnn_model.keras"
SCALER_PATH   = "models/tlkm_scaler.pkl"
DATA_PATH     = "data/data_tlkm_harga_saham.csv"
DB_PATH       = "predictions.db"
TEMPLATES_DIR = "templates"

TICKER       = "TLKM.JK"
WINDOW_SIZE  = 10
TARGET_COL   = "Close"
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RETURN_LAG_1", "RETURN_LAG_2",
    "RSI_SLOPE", "ROLL_STD_RETURN_5D",
    "MA_5", "MA_10",
]

# =============================================================================
# GLOBAL STATE
# =============================================================================
app_state: dict = {"model": None, "scaler": None}
_db_lock = threading.Lock()


# =============================================================================
# DATABASE
# =============================================================================
def init_db() -> None:
    with _db_lock:
        con = sqlite3.connect(DB_PATH)
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date   TEXT NOT NULL,
                    predicted_price   REAL NOT NULL,
                    last_actual_price REAL NOT NULL,
                    price_change      REAL NOT NULL,
                    price_change_pct  REAL NOT NULL,
                    trend             TEXT NOT NULL,
                    timestamp         TEXT NOT NULL
                )
            """)
            con.commit()
        finally:
            con.close()


def save_prediction(result: dict) -> int:
    with _db_lock:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(
                """INSERT INTO prediction_history
                   (prediction_date, predicted_price, last_actual_price,
                    price_change, price_change_pct, trend, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    result["prediction_date"],
                    result["predicted_price"],
                    result["last_actual_price"],
                    result["price_change"],
                    result["price_change_pct"],
                    result["trend"],
                    result["timestamp"],
                ),
            )
            con.commit()
            return cur.lastrowid
        finally:
            con.close()


def fetch_history(limit: int = 100) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT id, prediction_date, predicted_price, last_actual_price,
                      price_change, price_change_pct, trend, timestamp
               FROM prediction_history
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


# =============================================================================
# FEATURE ENGINEERING (identik dengan training)
# =============================================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("Kolom 'Close' tidak ditemukan")

    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    rsi_14   = 100 - (100 / (1 + rs))
    return_1d = df["Close"].pct_change()

    df["RETURN_LAG_1"]       = return_1d.shift(1)
    df["RETURN_LAG_2"]       = return_1d.shift(2)
    df["RSI_SLOPE"]          = rsi_14.diff()
    df["ROLL_STD_RETURN_5D"] = return_1d.rolling(window=5, min_periods=5).std()
    df["MA_5"]               = df["Close"].rolling(window=5,  min_periods=5).mean()
    df["MA_10"]              = df["Close"].rolling(window=10, min_periods=10).mean()

    df = df.dropna()
    return df[[
        "Open", "High", "Low", "Close", "Volume",
        "RETURN_LAG_1", "RETURN_LAG_2",
        "RSI_SLOPE", "ROLL_STD_RETURN_5D",
        "MA_5", "MA_10",
    ]]


# =============================================================================
# UPDATE DATA — ambil data terbaru dari Yahoo Finance (NO RETRAINING)
# =============================================================================
def update_latest_data() -> None:
    """
    Mengambil data harga terbaru TLKM.JK dari Yahoo Finance dan
    menggabungkannya ke file CSV lokal.

    Aturan ketat:
      - TIDAK memanggil model.fit() / scaler.fit() / model.save()
      - Hanya memperbarui raw OHLCV sebelum feature engineering
      - Aman dipanggil berulang kali (idempotent)
    """
    print("[update_latest_data] Memulai pembaruan data...")

    # ── 1. Pastikan file CSV lama ada ────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"[update_latest_data] SKIP — file tidak ditemukan: {DATA_PATH}")
        return

    try:
        # ── 2. Baca dataset lama ─────────────────────────────────────────
        df_old = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])
        df_old.index = pd.to_datetime(df_old.index, errors="coerce")
        df_old = df_old[~df_old.index.isna()].copy()
        df_old.index = pd.DatetimeIndex(df_old.index.normalize())   # strip time
        df_old = df_old.sort_index()

        last_date = df_old.index[-1]
        print(f"[update_latest_data] Tanggal terakhir dataset: {last_date.date()}")

        # ── 3. Tentukan rentang download ──────────────────────────────────
        # Mulai dari last_date (inklusif) agar overlap 1 hari
        # yfinance: start inklusif, end eksklusif → tambah 2 hari untuk hari ini
        fetch_start = last_date.strftime("%Y-%m-%d")
        fetch_end   = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        print(f"[update_latest_data] Mengambil data {fetch_start} → {fetch_end} dari Yahoo Finance...")

        # ── 4. Download dari Yahoo Finance ───────────────────────────────
        df_new = yf.download(
            tickers    = TICKER,
            start      = fetch_start,
            end        = fetch_end,
            auto_adjust= True,          # harga sudah adjusted
            progress   = False,
        )

        if df_new is None or df_new.empty:
            print("[update_latest_data] Tidak ada data baru dari Yahoo Finance — dataset tetap.")
            return

        # ── 5. Bersihkan kolom MultiIndex dari yfinance ───────────────────
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)

        # Pastikan kolom yang dibutuhkan ada
        needed = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in needed if c not in df_new.columns]
        if missing:
            print(f"[update_latest_data] Kolom hilang dari Yahoo Finance: {missing} — SKIP")
            return

        df_new = df_new[needed].copy()
        df_new.index = pd.DatetimeIndex(df_new.index.normalize())   # strip time
        df_new = df_new.sort_index()

        # ── 6. Gabung lama + baru, hapus duplikat ────────────────────────
        # ~keep="last"~ → jika tanggal sama, data dari yfinance (terbaru) menang
        df_combined = pd.concat([df_old[needed], df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined = df_combined.sort_index()

        # ── 7. Hitung jumlah baris baru ───────────────────────────────────
        n_new = len(df_combined) - len(df_old)
        if n_new <= 0:
            print("[update_latest_data] Tidak ada baris baru — dataset sudah terkini.")
            return

        print(f"[update_latest_data] {n_new} baris baru ditambahkan. Total: {len(df_combined)} baris.")

        # ── 8. Simpan ke CSV (timpa file lama) ────────────────────────────
        df_combined.to_csv(DATA_PATH)
        print(f"[update_latest_data] Dataset disimpan ke: {DATA_PATH}")
        print(f"[update_latest_data] Rentang data: {df_combined.index[0].date()} → {df_combined.index[-1].date()}")

    except Exception as exc:
        # Tangani semua error agar API tidak crash
        print(f"[update_latest_data] ERROR (diabaikan): {exc}")


# =============================================================================
# LOAD DATA
# =============================================================================
def load_feature_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].copy()
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    df_feat = create_features(df)
    if len(df_feat) < WINDOW_SIZE:
        raise ValueError(f"Data tidak cukup: {len(df_feat)} baris")
    return df_feat


# =============================================================================
# INFERENCE (NO FIT, NO SAVE)
# =============================================================================
def run_inference(model, scaler, df_feat: pd.DataFrame) -> dict:
    last_window        = df_feat[FEATURE_COLS].iloc[-WINDOW_SIZE:].values
    last_window_scaled = scaler.transform(last_window)          # NO FIT
    X_pred             = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))

    y_pred_scaled = float(model.predict(X_pred, verbose=0).flatten()[0])

    target_idx      = FEATURE_COLS.index(TARGET_COL)
    predicted_price = float(
        (y_pred_scaled - scaler.min_[target_idx]) / scaler.scale_[target_idx]
    )

    last_actual_date  = df_feat.index[-1]
    last_actual_price = float(df_feat[TARGET_COL].iloc[-1])
    prediction_date   = get_next_trading_day(last_actual_date)
    price_change      = predicted_price - last_actual_price
    price_change_pct  = (price_change / last_actual_price) * 100

    return {
        "predicted_price"  : round(predicted_price,   2),
        "prediction_date"  : prediction_date.strftime("%Y-%m-%d"),
        "prediction_day"   : prediction_date.strftime("%A"),
        "last_actual_date" : last_actual_date.strftime("%Y-%m-%d"),
        "last_actual_day"  : last_actual_date.strftime("%A"),
        "last_actual_price": round(last_actual_price, 2),
        "price_change"     : round(price_change,      2),
        "price_change_pct" : round(price_change_pct,  4),
        "trend"            : "bullish" if price_change > 0 else "bearish" if price_change < 0 else "flat",
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 TLKM Prediction API — Starting up...")
    print("=" * 60)

    for path, label in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler")]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"[ERROR] {label} tidak ditemukan: {path}\n"
                "Jalankan training dulu: python tlkm_rnn_main.py"
            )

    app_state["model"]  = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model  : {MODEL_PATH}")

    app_state["scaler"] = joblib.load(SCALER_PATH)
    print(f"  ✓ Scaler : {SCALER_PATH}")

    init_db()
    print(f"  ✓ DB     : {DB_PATH}")
    print(f"  ✓ Template dir: {os.path.abspath(TEMPLATES_DIR)}")

    # Cek apakah index.html ada
    tmpl_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(tmpl_path):
        print(f"  ✓ index.html ditemukan")
    else:
        print(f"  ✗ [WARNING] index.html TIDAK ditemukan di: {os.path.abspath(tmpl_path)}")

    print("=" * 60)
    print("  ✓ API siap → http://127.0.0.1:8000")
    print("=" * 60)

    yield

    app_state["model"]  = None
    app_state["scaler"] = None
    print("👋 API shutdown.")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title    = "TLKM Stock Price Prediction API",
    version  = "2.1.0",
    lifespan = lifespan,
)

# ── CORS: izinkan request dari browser (localhost) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =============================================================================
# SCHEMAS
# =============================================================================
class PredictionResponse(BaseModel):
    predicted_price  : float
    prediction_date  : str
    prediction_day   : str
    last_actual_date : str
    last_actual_day  : str
    last_actual_price: float
    price_change     : float
    price_change_pct : float
    trend            : str
    timestamp        : str
    saved_id         : int = Field(..., description="ID di SQLite")


class HistoryResponse(BaseModel):
    total  : int
    history: list[dict]


# =============================================================================
# ENDPOINT: GET /
# =============================================================================
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    """Halaman utama dashboard."""
    print("[GET /] Halaman utama diminta")
    return templates.TemplateResponse("index.html", {"request": request})


# =============================================================================
# ENDPOINT: GET /predict
# =============================================================================
@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_next_day():
    """
    Prediksi harga saham TLKM untuk next trading day.

    Pipeline lengkap:
      1. update_latest_data()   — ambil data terbaru dari Yahoo Finance
      2. load_feature_data()    — baca CSV + feature engineering
      3. run_inference()        — prediksi dengan model (NO FIT / NO SAVE)
      4. save_prediction()      — simpan hasil ke SQLite

    ⚠ Inference-only. Tidak ada model.fit(), scaler.fit(), model.save().
    """
    print("[GET /predict] ══════════════════════════════════════")
    print("[GET /predict] Request diterima — memulai pipeline...")

    # ── Cek model & scaler sudah ter-load ────────────────────────────────
    if app_state["model"] is None or app_state["scaler"] is None:
        print("[GET /predict] ERROR: model/scaler belum loaded")
        raise HTTPException(status_code=503, detail={
            "error": "Model belum ter-load",
            "detail": "Restart server dan tunggu hingga startup selesai.",
        })

    # ── STEP 1: Update data terbaru dari Yahoo Finance ───────────────────
    # Error ditangani di dalam fungsi — tidak akan crash API
    print("[GET /predict] STEP 1 — Update data dari Yahoo Finance...")
    update_latest_data()

    # ── STEP 2: Load & feature engineering ──────────────────────────────
    print("[GET /predict] STEP 2 — Load data & feature engineering...")
    try:
        df_feat = load_feature_data()
        print(f"[GET /predict] Data siap: {len(df_feat)} baris | "
              f"rentang: {df_feat.index[0].date()} → {df_feat.index[-1].date()}")
    except Exception as exc:
        print(f"[GET /predict] ERROR load data: {exc}")
        raise HTTPException(status_code=500, detail={
            "error": "Gagal memuat data", "detail": str(exc),
        }) from exc

    # ── STEP 3: Inference (NO FIT, NO SAVE) ──────────────────────────────
    print("[GET /predict] STEP 3 — Menjalankan inferensi model...")
    try:
        result = run_inference(app_state["model"], app_state["scaler"], df_feat)
        print(f"[GET /predict] Prediksi: Rp {result['predicted_price']:,.2f} "
              f"| {result['prediction_date']} ({result['trend'].upper()})")
    except Exception as exc:
        print(f"[GET /predict] ERROR inferensi: {exc}")
        raise HTTPException(status_code=500, detail={
            "error": "Inferensi gagal", "detail": str(exc),
        }) from exc

    # ── STEP 4: Simpan ke SQLite ──────────────────────────────────────────
    print("[GET /predict] STEP 4 — Menyimpan hasil ke database...")
    try:
        row_id = save_prediction(result)
        print(f"[GET /predict] Tersimpan dengan ID #{row_id}")
    except Exception as exc:
        print(f"[GET /predict] ERROR simpan DB: {exc}")
        raise HTTPException(status_code=500, detail={
            "error": "Gagal simpan ke database", "detail": str(exc),
        }) from exc

    print(f"[GET /predict] ✓ Selesai — ID #{row_id}")
    print("[GET /predict] ══════════════════════════════════════")
    return PredictionResponse(**result, saved_id=row_id)


# =============================================================================
# ENDPOINT: GET /history
# =============================================================================
@app.get("/history", response_model=HistoryResponse, tags=["History"])
def get_history(limit: int = 100):
    """Riwayat prediksi dari SQLite, terbaru di atas."""
    print(f"[GET /history] Request diterima, limit={limit}")

    try:
        rows = fetch_history(limit=limit)
        print(f"[GET /history] Mengembalikan {len(rows)} record")
        return HistoryResponse(total=len(rows), history=rows)
    except Exception as exc:
        print(f"[GET /history] ERROR: {exc}")
        raise HTTPException(status_code=500, detail={
            "error": "Gagal baca DB", "detail": str(exc),
        }) from exc


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)