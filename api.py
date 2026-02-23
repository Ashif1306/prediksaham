"""
api.py — TLKM Stock Prediction API v2.7.0
==========================================
Changelog v2.7.0:
  - Ganti range 1W → 5D (period=5d, interval=15m)
  - /chart-data mengembalikan data TANPA gap weekend/libur:
    data sudah di-sort dan diindex secara sequential (bukan datetime),
    sehingga frontend bisa pakai categorical axis dan tidak ada celah.
  - Field tambahan: 'labels' (list string tanggal/jam untuk x-axis label)
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
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tensorflow import keras
import joblib

from calendar_utils import get_next_trading_day


# =============================================================================
# KONFIGURASI
# =============================================================================
MODEL_PATH    = "models/tlkm_rnn_model.keras"
SCALER_PATH   = "models/tlkm_scaler.pkl"
DATA_PATH     = "data/data_tlkm_harga_saham.csv"
DB_PATH       = "predictions.db"
TEMPLATES_DIR = "templates"

WINDOW_SIZE  = 10
TARGET_COL   = "Close"
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RETURN_LAG_1", "RETURN_LAG_2",
    "RSI_SLOPE", "ROLL_STD_RETURN_5D",
    "MA_5", "MA_10",
]
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Range config — 1W diganti 5D
CHART_RANGE_CONFIG = {
    "1D" : {"period": "1d",  "interval": "1m" },
    "5D" : {"period": "5d",  "interval": "15m"},
    "1M" : {"period": "1mo", "interval": "1d" },
    "YTD": {"period": "ytd", "interval": "1d" },
    "1Y" : {"period": "1y",  "interval": "1d" },
    "3Y" : {"period": "3y",  "interval": "1wk"},
}

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
                    prediction_date   TEXT NOT NULL UNIQUE,
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


def upsert_prediction(result: dict) -> tuple[int, bool]:
    with _db_lock:
        con = sqlite3.connect(DB_PATH)
        try:
            row = con.execute(
                "SELECT id FROM prediction_history WHERE prediction_date = ?",
                (result["prediction_date"],)
            ).fetchone()
            if row:
                con.execute(
                    """UPDATE prediction_history SET
                         predicted_price=?, last_actual_price=?,
                         price_change=?, price_change_pct=?,
                         trend=?, timestamp=?
                       WHERE prediction_date=?""",
                    (result["predicted_price"], result["last_actual_price"],
                     result["price_change"], result["price_change_pct"],
                     result["trend"], result["timestamp"],
                     result["prediction_date"]),
                )
                con.commit()
                return row[0], False
            else:
                cur = con.execute(
                    """INSERT INTO prediction_history
                       (prediction_date, predicted_price, last_actual_price,
                        price_change, price_change_pct, trend, timestamp)
                       VALUES (?,?,?,?,?,?,?)""",
                    (result["prediction_date"], result["predicted_price"],
                     result["last_actual_price"], result["price_change"],
                     result["price_change_pct"], result["trend"],
                     result["timestamp"]),
                )
                con.commit()
                return cur.lastrowid, True
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
               ORDER BY prediction_date DESC LIMIT ?""", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def delete_prediction_by_id(record_id: int) -> bool:
    with _db_lock:
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(
                "DELETE FROM prediction_history WHERE id = ?", (record_id,))
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()


# =============================================================================
# HELPER
# =============================================================================
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()].copy()
    df.index = pd.DatetimeIndex(df.index)
    for col in OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


# =============================================================================
# UPDATE DATA
# =============================================================================
def update_latest_data() -> None:
    print("[update] ══════════════════════════════════════════════════════")
    print(f"[update] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not os.path.exists(DATA_PATH):
        print(f"[update] ⚠  File tidak ada — dilewati"); return
    try:
        df_old = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        df_old = _normalize_df(df_old)
    except Exception as exc:
        print(f"[update] ⚠  Gagal baca CSV: {exc} — dilewati"); return
    if df_old.empty:
        print("[update] ⚠  Dataset kosong — dilewati"); return

    last_date   = pd.Timestamp(df_old.index.max()).normalize()
    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[update] Last: {last_date.strftime('%Y-%m-%d %A')} | Fetch: {fetch_start}→{fetch_end}")

    try:
        df_new = yf.download(
            tickers="TLKM.JK", start=fetch_start, end=fetch_end,
            interval="1d", progress=False, auto_adjust=False,
        )
    except Exception as exc:
        print(f"[update] ⚠  Yahoo error: {exc}")
        print("[update] ══════════════════════════════════════════════════════"); return

    print(f"[update] Download: {len(df_new)} baris")
    if df_new is None or df_new.empty:
        print("[update] ✓  Tidak ada data baru.")
        print("[update] ══════════════════════════════════════════════════════"); return

    tmp_cols = (df_new.columns.get_level_values(0)
                if isinstance(df_new.columns, pd.MultiIndex) else df_new.columns)
    keep = [c for c in OHLCV_COLS if c in tmp_cols]
    df_new = _normalize_df(df_new[keep].copy())
    df_new = df_new[df_new.index > last_date].copy()

    if df_new.empty:
        print("[update] ✓  Semua sudah ada.")
        print("[update] ══════════════════════════════════════════════════════"); return

    combined = pd.concat([df_old, df_new], axis=0)
    combined = _normalize_df(combined)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    try:
        combined.to_csv(DATA_PATH)
        print(f"[update] ✓  +{len(df_new)} baris | last={pd.Timestamp(combined.index.max()).strftime('%Y-%m-%d %A')}")
    except Exception as exc:
        print(f"[update] ⚠  Gagal simpan: {exc}")
    print("[update] ══════════════════════════════════════════════════════")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Kolom 'Close' tidak ditemukan")
    df = df.copy()
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rsi_14   = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
    ret1d    = df["Close"].pct_change()
    df["RETURN_LAG_1"]       = ret1d.shift(1)
    df["RETURN_LAG_2"]       = ret1d.shift(2)
    df["RSI_SLOPE"]          = rsi_14.diff()
    df["ROLL_STD_RETURN_5D"] = ret1d.rolling(5, min_periods=5).std()
    df["MA_5"]               = df["Close"].rolling(5,  min_periods=5).mean()
    df["MA_10"]              = df["Close"].rolling(10, min_periods=10).mean()
    feat_cols = ["RETURN_LAG_1","RETURN_LAG_2","RSI_SLOPE","ROLL_STD_RETURN_5D","MA_5","MA_10"]
    return df.dropna(subset=feat_cols)[FEATURE_COLS]


def load_and_prepare_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = _normalize_df(df)
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLCV tidak ditemukan: {missing}")
    df = df[OHLCV_COLS].copy()
    df = df.dropna(subset=OHLCV_COLS)
    df_feat = create_features(df)
    if len(df_feat) < WINDOW_SIZE:
        raise ValueError(f"Data tidak cukup: {len(df_feat)} baris")
    print(f"[load] {len(df_feat)} baris | last={df_feat.index[-1].strftime('%Y-%m-%d %A')} | Close=Rp {df_feat['Close'].iloc[-1]:,.2f}")
    return df_feat


# =============================================================================
# INFERENCE
# =============================================================================
def predict_next_trading_day(model, scaler, df_feat: pd.DataFrame) -> dict:
    last_window        = df_feat[FEATURE_COLS].iloc[-WINDOW_SIZE:].values
    last_window_scaled = scaler.transform(last_window)
    X_pred             = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
    y_pred_scaled      = float(model.predict(X_pred, verbose=0).flatten()[0])
    target_idx         = FEATURE_COLS.index(TARGET_COL)
    predicted_price    = float((y_pred_scaled - scaler.min_[target_idx]) / scaler.scale_[target_idx])
    last_actual_date   = df_feat.index[-1]
    last_actual_price  = float(df_feat[TARGET_COL].iloc[-1])
    prediction_date    = get_next_trading_day(last_actual_date)
    price_change       = predicted_price - last_actual_price
    price_change_pct   = (price_change / last_actual_price) * 100
    return {
        "predicted_price"  : round(predicted_price,   2),
        "prediction_date"  : prediction_date.strftime("%Y-%m-%d"),
        "prediction_day"   : prediction_date.strftime("%A"),
        "last_actual_date" : last_actual_date.strftime("%Y-%m-%d"),
        "last_actual_day"  : last_actual_date.strftime("%A"),
        "last_actual_price": round(last_actual_price, 2),
        "price_change"     : round(price_change,      2),
        "price_change_pct" : round(price_change_pct,  4),
        "trend"            : ("bullish" if price_change > 0 else "bearish" if price_change < 0 else "flat"),
        "timestamp"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 TLKM Prediction API v2.7.0 — Starting up...")
    print("=" * 60)
    for path, label in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler")]:
        if not os.path.exists(path):
            raise RuntimeError(f"[ERROR] {label} tidak ditemukan: {path}")
    app_state["model"]  = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model  : {MODEL_PATH}")
    app_state["scaler"] = joblib.load(SCALER_PATH)
    print(f"  ✓ Scaler : {SCALER_PATH}")
    init_db()
    print(f"  ✓ DB     : {DB_PATH}")
    print(f"  {'✓' if os.path.exists(os.path.join(TEMPLATES_DIR,'index.html')) else '✗'} index.html")
    print("=" * 60)
    yield
    app_state["model"] = None; app_state["scaler"] = None
    print("👋 Shutdown.")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(title="TLKM Stock Prediction API", version="2.7.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","DELETE"], allow_headers=["*"])
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =============================================================================
# SCHEMAS
# =============================================================================
class PredictionResponse(BaseModel):
    predicted_price: float; prediction_date: str; prediction_day: str
    last_actual_date: str; last_actual_day: str; last_actual_price: float
    price_change: float; price_change_pct: float; trend: str
    timestamp: str; saved_id: int; is_new_record: bool

class HistoryResponse(BaseModel):
    total: int; history: list[dict]

class DeleteResponse(BaseModel):
    success: bool; message: str

class RecentPriceItem(BaseModel):
    date: str; day: str; open: float; high: float; low: float; close: float; volume: float

class RecentPricesResponse(BaseModel):
    total: int; prices: list[RecentPriceItem]

class ChartDataResponse(BaseModel):
    range   : str
    ticker  : str
    labels  : list[str]   # label untuk x-axis (sequential, no gaps)
    candles : list[dict]  # [{o,h,l,c}] — index sequential, bukan datetime
    volumes : list[dict]  # [{y, color}]
    latest  : dict

class StatusResponse(BaseModel):
    csv_last_date: str; csv_total_rows: int; feat_last_date: str
    feat_total_rows: int; next_trading_day: str; model_loaded: bool
    scaler_loaded: bool; model_path_exists: bool; scaler_path_exists: bool
    csv_path_exists: bool; server_time: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_next_day():
    print(f"\n[/predict] ══ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ══")
    if app_state["model"] is None or app_state["scaler"] is None:
        raise HTTPException(status_code=503, detail={"error": "Model belum ter-load"})
    update_latest_data()
    try:
        df_feat = load_and_prepare_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Gagal load data", "detail": str(exc)}) from exc
    try:
        result = predict_next_trading_day(app_state["model"], app_state["scaler"], df_feat)
        print(f"[/predict] → {result['prediction_date']}  Rp {result['predicted_price']:,.2f}  [{result['trend'].upper()}]")
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Inferensi gagal", "detail": str(exc)}) from exc
    try:
        row_id, is_new = upsert_prediction(result)
        print(f"[/predict] DB #{row_id}  ({'INSERT' if is_new else 'UPDATE'})")
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Gagal simpan DB", "detail": str(exc)}) from exc
    return PredictionResponse(**result, saved_id=row_id, is_new_record=is_new)


@app.get("/history", response_model=HistoryResponse, tags=["History"])
def get_history(limit: int = 100):
    try:
        return HistoryResponse(total=len(rows := fetch_history(limit=limit)), history=rows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.delete("/delete/{record_id}", response_model=DeleteResponse, tags=["History"])
def delete_record(record_id: int):
    try:
        if not delete_prediction_by_id(record_id):
            raise HTTPException(status_code=404, detail={"error": f"Record #{record_id} tidak ditemukan"})
        print(f"[/delete] #{record_id} dihapus")
        return DeleteResponse(success=True, message=f"Record #{record_id} berhasil dihapus")
    except HTTPException: raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/recent-prices", response_model=RecentPricesResponse, tags=["Data"])
def get_recent_prices(n: int = 10):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail={"error": f"File tidak ditemukan: {DATA_PATH}"})
    try:
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        df = _normalize_df(df)[OHLCV_COLS].dropna(subset=OHLCV_COLS)
        prices = [
            RecentPriceItem(
                date=dt.strftime("%Y-%m-%d"), day=dt.strftime("%A"),
                open=round(float(row["Open"]),2), high=round(float(row["High"]),2),
                low=round(float(row["Low"]),2),   close=round(float(row["Close"]),2),
                volume=round(float(row["Volume"]),0),
            ) for dt, row in df.tail(n).iloc[::-1].iterrows()
        ]
        return RecentPricesResponse(total=len(prices), prices=prices)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


# =============================================================================
# ENDPOINT: GET /chart-data — No-gap sequential data
# =============================================================================
@app.get("/chart-data", response_model=ChartDataResponse, tags=["Chart"])
def get_chart_data(
    range: str = Query(default="1M", description="Range: 1D | 5D | 1M | YTD | 1Y | 3Y")
):
    """
    Data OHLCV untuk candlestick chart.

    PENTING — No-gap design:
    Data dikembalikan secara SEQUENTIAL (bukan datetime axis).
    Field 'labels' berisi string tanggal/jam untuk tiap candle.
    Frontend menggunakan categorical axis sehingga weekend/libur
    tidak membuat celah/gap di chart.

    Range & konfigurasi:
      1D  → period=1d,  interval=1m   (intraday per menit)
      5D  → period=5d,  interval=15m  (5 hari trading, per 15 menit)
      1M  → period=1mo, interval=1d   (bulanan, harian)
      YTD → period=ytd, interval=1d   (year-to-date, harian)
      1Y  → period=1y,  interval=1d   (tahunan, harian)
      3Y  → period=3y,  interval=1wk  (3 tahun, mingguan)
    """
    range_upper = range.upper()
    if range_upper not in CHART_RANGE_CONFIG:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Range tidak valid: '{range}'. Gunakan: {list(CHART_RANGE_CONFIG.keys())}"}
        )

    cfg = CHART_RANGE_CONFIG[range_upper]
    print(f"[/chart-data] range={range_upper} period={cfg['period']} interval={cfg['interval']}")

    try:
        df = yf.download(
            tickers="TLKM.JK",
            period=cfg["period"],
            interval=cfg["interval"],
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail={
            "error": "Gagal ambil data dari Yahoo Finance", "detail": str(exc)}) from exc

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail={"error": f"Tidak ada data untuk range={range_upper}"})

    # Normalisasi
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"]).sort_index()

    if df.empty:
        raise HTTPException(status_code=404, detail={"error": "Data kosong setelah normalisasi"})

    # Format label — intraday tampilkan jam, daily/weekly tampilkan tanggal
    is_intraday = cfg["interval"] in ("1m","5m","15m","30m","1h")

    def fmt_label(dt: pd.Timestamp) -> str:
        if is_intraday:
            # Format: "19 Feb 10:30" — ringkas untuk intraday
            return dt.strftime("%-d %b %H:%M") if hasattr(dt, 'strftime') else str(dt)
        return dt.strftime("%-d %b '%y") if cfg["interval"] == "1wk" else dt.strftime("%-d %b")

    # Build sequential arrays (NO datetime — no gaps!)
    labels  = []
    candles = []
    volumes = []

    for dt, row in df.iterrows():
        try:
            label = fmt_label(dt)
        except Exception:
            label = str(dt)[:16]

        o  = round(float(row["Open"]),   2)
        h  = round(float(row["High"]),   2)
        lo = round(float(row["Low"]),    2)
        c  = round(float(row["Close"]),  2)
        v  = round(float(row.get("Volume", 0) or 0), 0)

        labels.append(label)
        candles.append({"o": o, "h": h, "l": lo, "c": c})
        volumes.append({"y": v, "color": "#0af5b0" if c >= o else "#ff4d6d"})

    # Latest info untuk header
    first_c = candles[0]["c"]
    last_c  = candles[-1]["c"]
    chg     = round(last_c - first_c, 2)
    chg_pct = round((chg / first_c) * 100, 2) if first_c else 0

    latest = {
        "close"     : last_c,
        "change"    : chg,
        "change_pct": chg_pct,
        "high"      : round(float(df["High"].max()),  2),
        "low"       : round(float(df["Low"].min()),   2),
        "date"      : labels[-1],
    }

    print(f"[/chart-data] ✓ {len(candles)} candles | last={latest['date']} close={last_c}")

    return ChartDataResponse(
        range=range_upper, ticker="TLKM.JK",
        labels=labels, candles=candles, volumes=volumes, latest=latest,
    )


@app.get("/status", response_model=StatusResponse, tags=["Debug"])
def get_status():
    csv_last_date = "N/A"; csv_total_rows = 0
    if os.path.exists(DATA_PATH):
        try:
            df_csv = _normalize_df(pd.read_csv(DATA_PATH, index_col=0, parse_dates=True))
            csv_total_rows = len(df_csv)
            if not df_csv.empty:
                csv_last_date = pd.Timestamp(df_csv.index.max()).strftime("%Y-%m-%d %A")
        except Exception as exc:
            csv_last_date = f"ERROR: {exc}"
    feat_last_date = "N/A"; feat_total_rows = 0; next_td = "N/A"
    if os.path.exists(DATA_PATH):
        try:
            df_feat = load_and_prepare_data(); feat_total_rows = len(df_feat)
            if not df_feat.empty:
                last_dt = df_feat.index[-1]
                feat_last_date = last_dt.strftime("%Y-%m-%d %A")
                next_td = get_next_trading_day(last_dt).strftime("%Y-%m-%d %A")
        except Exception as exc:
            feat_last_date = f"ERROR: {exc}"
    return StatusResponse(
        csv_last_date=csv_last_date, csv_total_rows=csv_total_rows,
        feat_last_date=feat_last_date, feat_total_rows=feat_total_rows,
        next_trading_day=next_td,
        model_loaded=app_state["model"] is not None,
        scaler_loaded=app_state["scaler"] is not None,
        model_path_exists=os.path.exists(MODEL_PATH),
        scaler_path_exists=os.path.exists(SCALER_PATH),
        csv_path_exists=os.path.exists(DATA_PATH),
        server_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)