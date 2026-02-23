"""
predict_next_day.py
===================
Sistem prediksi harga saham TLKM untuk hari perdagangan berikutnya (next trading day).

Modul ini menggunakan model SimpleRNN yang sudah di-training untuk memprediksi
harga penutupan (Close) TLKM pada hari perdagangan berikutnya dengan mempertimbangkan
kalender libur Bursa Efek Indonesia (BEI).

Pipeline Prediksi:
1. Load model dan scaler yang sudah di-training
2. Update data terbaru dari Yahoo Finance (sync dengan api.py)
3. Load data historis terbaru
4. Ekstraksi 10 data terakhir (window size)
5. Normalisasi menggunakan scaler lama (NO REFIT!)
6. Reshape ke format RNN (1, 10, jumlah_fitur)
7. Model inference
8. Inverse transform untuk mendapatkan harga asli
9. Tentukan tanggal next trading day (skip weekend & holiday BEI)
10. Output hasil prediksi

CHANGELOG:
- Ditambahkan update_latest_data() sebelum prediksi (sync dengan api.py)
- _normalize_df() kini identik dengan api.py (timezone stripping, sort index)
- load_and_prepare_data() menggunakan pipeline yang sama dengan api.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

# Import library ML
from tensorflow import keras
import joblib
import yfinance as yf

# Import utility kalender
from calendar_utils import get_next_trading_day, is_trading_day


# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

# Jalankan script dari root folder project agar path relatif berikut valid.
MODEL_PATH  = 'models/tlkm_rnn_model.keras'
SCALER_PATH = 'models/tlkm_scaler.pkl'
DATA_PATH   = 'data/data_tlkm_harga_saham.csv'
PREDICTION_OUTPUT_PATH = 'predictions/latest_prediction.json'

# Konfigurasi model (HARUS SAMA dengan training)
WINDOW_SIZE  = 10
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RETURN_LAG_1', 'RETURN_LAG_2',
    'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
    'MA_5', 'MA_10'
]
TARGET_COL   = 'Close'
OHLCV_COLS   = ['Open', 'High', 'Low', 'Close', 'Volume']


# =============================================================================
# HELPER — identik dengan api.py
# =============================================================================

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi DataFrame: flatten MultiIndex kolom, strip timezone,
    konversi numerik, dan sort index.
    Identik dengan _normalize_df() di api.py.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    # Strip timezone agar tidak ada mismatch timezone — ini penyebab utama
    # perbedaan tanggal antara predict_next_day.py dan api.py
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()].copy()
    df.index = pd.DatetimeIndex(df.index)
    for col in OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_index()


# =============================================================================
# UPDATE DATA — identik dengan api.py
# =============================================================================

def update_latest_data() -> None:
    """
    Unduh data terbaru dari Yahoo Finance dan append ke CSV lokal.
    Identik dengan update_latest_data() di api.py.
    Dipanggil sebelum prediksi agar data selalu up-to-date.
    """
    print("\n[update] ══════════════════════════════════════════════════════")
    print(f"[update] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(DATA_PATH):
        print(f"[update] ⚠  File tidak ada — dilewati")
        return

    try:
        df_old = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        df_old = _normalize_df(df_old)
    except Exception as exc:
        print(f"[update] ⚠  Gagal baca CSV: {exc} — dilewati")
        return

    if df_old.empty:
        print("[update] ⚠  Dataset kosong — dilewati")
        return

    last_date   = pd.Timestamp(df_old.index.max()).normalize()
    fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    fetch_end   = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"[update] Last: {last_date.strftime('%Y-%m-%d %A')} | Fetch: {fetch_start}→{fetch_end}")

    try:
        df_new = yf.download(
            tickers='TLKM.JK',
            start=fetch_start,
            end=fetch_end,
            interval='1d',
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        print(f"[update] ⚠  Yahoo error: {exc}")
        print("[update] ══════════════════════════════════════════════════════")
        return

    print(f"[update] Download: {len(df_new)} baris")

    if df_new is None or df_new.empty:
        print("[update] ✓  Tidak ada data baru.")
        print("[update] ══════════════════════════════════════════════════════")
        return

    tmp_cols = (df_new.columns.get_level_values(0)
                if isinstance(df_new.columns, pd.MultiIndex) else df_new.columns)
    keep   = [c for c in OHLCV_COLS if c in tmp_cols]
    df_new = _normalize_df(df_new[keep].copy())
    df_new = df_new[df_new.index > last_date].copy()

    if df_new.empty:
        print("[update] ✓  Semua sudah ada.")
        print("[update] ══════════════════════════════════════════════════════")
        return

    combined = pd.concat([df_old, df_new], axis=0)
    combined = _normalize_df(combined)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    try:
        combined.to_csv(DATA_PATH)
        print(f"[update] ✓  +{len(df_new)} baris | last={pd.Timestamp(combined.index.max()).strftime('%Y-%m-%d %A')}")
    except Exception as exc:
        print(f"[update] ⚠  Gagal simpan: {exc}")

    print("[update] ══════════════════════════════════════════════════════")


# =============================================================================
# FUNGSI FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'Close' not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan dalam dataset")

    df = df.copy()

    # ── Intermediate: RSI-14 ───────────────────────────────────────────────
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    rsi_14   = 100 - (100 / (1 + rs))

    # ── Intermediate: Return harian ────────────────────────────────────────
    return_1d = df['Close'].pct_change()

    # ── 6 Fitur Teknikal Final ─────────────────────────────────────────────
    df['RETURN_LAG_1']       = return_1d.shift(1)
    df['RETURN_LAG_2']       = return_1d.shift(2)
    df['RSI_SLOPE']          = rsi_14.diff()
    df['ROLL_STD_RETURN_5D'] = return_1d.rolling(window=5, min_periods=5).std()
    df['MA_5']               = df['Close'].rolling(window=5,  min_periods=5).mean()
    df['MA_10']              = df['Close'].rolling(window=10, min_periods=10).mean()

    df = df.dropna()

    FINAL_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RETURN_LAG_1', 'RETURN_LAG_2',
        'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
        'MA_5', 'MA_10'
    ]
    return df[FINAL_COLS]


# =============================================================================
# FUNGSI LOAD MODEL & DATA — load_and_prepare_data() identik dengan api.py
# =============================================================================

def load_model_and_scaler():
    """
    Load model dan scaler yang sudah di-training.
    """
    print("\n📦 Loading model dan scaler...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            f"Pastikan Anda sudah menjalankan training terlebih dahulu!"
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler tidak ditemukan: {SCALER_PATH}\n"
            f"Pastikan Anda sudah menjalankan training terlebih dahulu!"
        )

    model  = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model loaded  : {MODEL_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✓ Scaler loaded : {SCALER_PATH}")

    return model, scaler


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load data historis dan lakukan feature engineering.
    Pipeline identik dengan load_and_prepare_data() di api.py:
      - Gunakan _normalize_df() (timezone stripping, sort index)
      - Hanya ambil kolom OHLCV sebelum feature engineering
      - dropna berdasarkan OHLCV_COLS
    """
    print("\n📊 Loading data historis...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data tidak ditemukan: {DATA_PATH}\n"
            f"Pastikan Anda sudah menjalankan training untuk generate data!"
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # ── Gunakan _normalize_df() — identik dengan api.py ───────────────────
    df = _normalize_df(df)

    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom OHLCV tidak ditemukan: {missing}")

    df = df[OHLCV_COLS].copy()
    df = df.dropna(subset=OHLCV_COLS)

    if df.empty:
        raise ValueError("Dataset kosong setelah normalisasi.")

    print(f"  ✓ Data loaded : {len(df)} baris")
    print(f"  ✓ Periode     : {df.index[0].date()} hingga {df.index[-1].date()}")

    # ── Feature engineering ────────────────────────────────────────────────
    print("\n🔧 Melakukan feature engineering...")
    df_feat = create_features(df)

    if len(df_feat) < WINDOW_SIZE:
        raise ValueError(f"Data tidak cukup: {len(df_feat)} baris (butuh min {WINDOW_SIZE})")

    missing_features = [f for f in FEATURE_COLS if f not in df_feat.columns]
    if missing_features:
        raise ValueError(f"Fitur tidak ditemukan: {missing_features}")

    print(f"  ✓ Features created : {len(df_feat)} baris (setelah dropna)")
    print(f"  ✓ Kolom df_feat    : {df_feat.shape[1]} ← tepat 11, tidak ada intermediate")
    print(f"  ✓ Last data        : {df_feat.index[-1].strftime('%Y-%m-%d %A')} | Close=Rp {df_feat['Close'].iloc[-1]:,.2f}")

    return df_feat


# =============================================================================
# FUNGSI PREDIKSI UTAMA
# =============================================================================

def predict_next_trading_day(model, scaler, df_recent, verbose=True):
    if verbose:
        print("\n" + "="*70)
        print("PREDIKSI NEXT TRADING DAY")
        print("="*70)

    # ── STEP 1: Validasi ───────────────────────────────────────────────────
    if len(df_recent) < WINDOW_SIZE:
        raise ValueError(
            f"Data tidak cukup. Butuh minimal {WINDOW_SIZE} baris, "
            f"tersedia: {len(df_recent)}"
        )
    if verbose:
        print(f"\n✓ Validasi data: OK ({len(df_recent)} baris tersedia)")

    # ── STEP 2: Ekstraksi window ───────────────────────────────────────────
    last_window = df_recent[FEATURE_COLS].iloc[-WINDOW_SIZE:].values
    if last_window.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
        raise ValueError(
            f"Shape input sebelum scaling tidak sesuai. "
            f"Expected {(WINDOW_SIZE, len(FEATURE_COLS))}, got {last_window.shape}"
        )
    if verbose:
        print(f"\n📊 Data Input:")
        print(f"  Window size      : {WINDOW_SIZE} hari")
        print(f"  Jumlah fitur     : {len(FEATURE_COLS)}")
        print(f"  Tanggal terakhir : {df_recent.index[-1].date()}")
        print(f"  Harga terakhir   : Rp {df_recent[TARGET_COL].iloc[-1]:,.2f}")

    # ── STEP 3: Normalisasi (NO REFIT!) ────────────────────────────────────
    scaler_n_features = getattr(scaler, 'n_features_in_', None)
    if scaler_n_features is None:
        scaler_n_features = len(getattr(scaler, 'scale_', []))
    if scaler_n_features != len(FEATURE_COLS):
        raise ValueError(
            f"Jumlah fitur tidak cocok dengan scaler training. "
            f"Scaler expects {scaler_n_features}, pipeline memberikan {len(FEATURE_COLS)}."
        )
    last_window_scaled = scaler.transform(last_window)
    if verbose:
        print(f"\n🔧 Preprocessing:")
        print(f"  Normalisasi      : MinMaxScaler (dari training)")
        print(f"  Shape sebelum    : {last_window.shape}")
        print(f"  Shape sesudah    : {last_window_scaled.shape}")

    # ── STEP 4: Reshape untuk RNN ──────────────────────────────────────────
    X_pred = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))
    if verbose:
        print(f"  Reshape untuk RNN: {X_pred.shape}")

    # ── STEP 5: Inference ──────────────────────────────────────────────────
    y_pred_scaled = float(model.predict(X_pred, verbose=0).flatten()[0])
    if verbose:
        print(f"\n🤖 Model Inference:")
        print(f"  Output (scaled)  : {y_pred_scaled:.6f}")

    # ── STEP 6: Inverse transform ──────────────────────────────────────────
    target_idx      = FEATURE_COLS.index(TARGET_COL)
    predicted_price = float(
        (y_pred_scaled - scaler.min_[target_idx]) / scaler.scale_[target_idx]
    )
    if verbose:
        print(f"  Inverse transform: Rp {predicted_price:,.2f}")

    # ── STEP 7 & 8: Tanggal prediksi ───────────────────────────────────────
    last_actual_date  = df_recent.index[-1]
    last_actual_price = float(df_recent[TARGET_COL].iloc[-1])
    prediction_date   = get_next_trading_day(last_actual_date)
    price_change      = predicted_price - last_actual_price
    price_change_pct  = (price_change / last_actual_price) * 100

    if verbose:
        print(f"\n📅 Trading Day Calculation:")
        print(f"  Last data date   : {last_actual_date.date()} ({last_actual_date.strftime('%A')})")
        print(f"  Prediction date  : {prediction_date.date()} ({prediction_date.strftime('%A')})")
        days_diff = (prediction_date - last_actual_date).days
        if days_diff > 1:
            print(f"  Days skipped     : {days_diff - 1} hari")
            print(f"    └─ Alasan: Weekend/Holiday BEI")

    # ── STEP 9: Output ─────────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*70)
        print("📈 HASIL PREDIKSI")
        print("="*70)
        print(f"  Tanggal Prediksi : {prediction_date.strftime('%A, %d %B %Y')}")
        print(f"  Harga Prediksi   : Rp {predicted_price:,.2f}")
        print(f"  Harga Terakhir   : Rp {last_actual_price:,.2f}")
        print(f"  Perubahan        : Rp {price_change:+,.2f} ({price_change_pct:+.2f}%)")
        if price_change > 0:
            trend = "📈 NAIK (Bullish)"
        elif price_change < 0:
            trend = "📉 TURUN (Bearish)"
        else:
            trend = "➡️ FLAT (Neutral)"
        print(f"  Tren Prediksi    : {trend}")
        print("="*70)

    return {
        'predicted_price'  : float(predicted_price),
        'prediction_date'  : prediction_date,
        'last_actual_date' : last_actual_date,
        'last_actual_price': float(last_actual_price),
        'price_change'     : float(price_change),
        'price_change_pct' : float(price_change_pct),
    }


# =============================================================================
# FUNGSI SAVE HASIL PREDIKSI
# =============================================================================

def save_prediction_result(result: dict) -> str:
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT_PATH), exist_ok=True)

    output = {
        'predicted_price'  : result['predicted_price'],
        'prediction_date'  : result['prediction_date'].strftime('%Y-%m-%d'),
        'prediction_day'   : result['prediction_date'].strftime('%A'),
        'last_actual_date' : result['last_actual_date'].strftime('%Y-%m-%d'),
        'last_actual_day'  : result['last_actual_date'].strftime('%A'),
        'last_actual_price': result['last_actual_price'],
        'price_change'     : result['price_change'],
        'price_change_pct' : result['price_change_pct'],
        'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(PREDICTION_OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    return PREDICTION_OUTPUT_PATH


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SISTEM PREDIKSI HARGA SAHAM TLKM")
    print("Next Trading Day Prediction")
    print("="*70)

    try:
        model, scaler = load_model_and_scaler()

        # ── UPDATE DATA SEBELUM PREDIKSI (sync dengan api.py) ───────────────
        update_latest_data()

        df_feat = load_and_prepare_data()

        result = predict_next_trading_day(
            model=model, scaler=scaler, df_recent=df_feat, verbose=True
        )

        output_path = save_prediction_result(result)
        print(f"\n💾 Hasil prediksi disimpan: {output_path}")
        print("\n" + "="*70)
        print("✅ PREDIKSI SELESAI")
        print("="*70)

        return result

    except FileNotFoundError as e:
        print("\n❌ ERROR: File tidak ditemukan!")
        print(f"   {e}")
        print("\nPastikan Anda sudah menjalankan training terlebih dahulu:")
        print("   python tlkm_stock_prediction_rnn.py")
        sys.exit(1)

    except ValueError as e:
        print("\n❌ ERROR: Validasi data gagal!")
        print(f"   {e}")
        sys.exit(1)

    except Exception as e:
        print("\n❌ ERROR: Terjadi kesalahan tidak terduga!")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()