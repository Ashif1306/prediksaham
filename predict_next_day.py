"""
predict_next_day.py
===================
Sistem prediksi harga saham TLKM untuk hari perdagangan berikutnya (next trading day).

Modul ini menggunakan model SimpleRNN yang sudah di-training untuk memprediksi
harga penutupan (Close) TLKM pada hari perdagangan berikutnya dengan mempertimbangkan
kalender libur Bursa Efek Indonesia (BEI).

Pipeline Prediksi:
1. Load model dan scaler yang sudah di-training
2. Load data historis terbaru
3. Ekstraksi 10 data terakhir (window size)
4. Normalisasi menggunakan scaler lama (NO REFIT!)
5. Reshape ke format RNN (1, 10, jumlah_fitur)
6. Model inference
7. Inverse transform untuk mendapatkan harga asli
8. Tentukan tanggal next trading day (skip weekend & holiday BEI)
9. Output hasil prediksi
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

# Import utility kalender
from calendar_utils import get_next_trading_day, is_trading_day


# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

# Jalankan script dari root folder project agar path relatif berikut valid.
MODEL_PATH = 'models/tlkm_rnn_model.keras'
SCALER_PATH = 'models/tlkm_scaler.pkl'
DATA_PATH = 'data/data_tlkm_harga_saham.csv'
PREDICTION_OUTPUT_PATH = 'predictions/latest_prediction.json'

# Konfigurasi model (HARUS SAMA dengan training)
WINDOW_SIZE = 10
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RETURN_LAG_1', 'RETURN_LAG_2', 
    'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
    'MA_5', 'MA_10'
]
TARGET_COL = 'Close'


# =============================================================================
# FUNGSI FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# =============================================================================

def create_features(df):
    """
    Membuat fitur teknikal untuk model regresi.

    PENTING: Fungsi ini HARUS identik dengan fungsi yang digunakan saat training
    untuk memastikan konsistensi fitur.

    Kolom intermediate (RSI_14, RETURN_1D, dll.) dihitung sebagai variabel
    lokal Python — tidak disimpan ke DataFrame — sehingga df yang dikembalikan
    hanya berisi 11 kolom final FEATURE_COLS.

    Args:
        df (pd.DataFrame): DataFrame dengan kolom OHLCV.

    Returns:
        pd.DataFrame: DataFrame dengan tepat 11 kolom final.
    """
    df = df.copy()

    if 'Close' not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan dalam dataset")

    # ── Intermediate: RSI-14 (hanya dibutuhkan untuk RSI_SLOPE) ───────────
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    rsi_14   = 100 - (100 / (1 + rs))          # variabel lokal, tidak ke df

    # ── Intermediate: Return harian (untuk LAG & ROLL_STD) ────────────────
    return_1d = df['Close'].pct_change()        # variabel lokal, tidak ke df

    # ── 6 Fitur Teknikal Final ─────────────────────────────────────────────
    df['RETURN_LAG_1']       = return_1d.shift(1)
    df['RETURN_LAG_2']       = return_1d.shift(2)
    df['RSI_SLOPE']          = rsi_14.diff()
    df['ROLL_STD_RETURN_5D'] = return_1d.rolling(window=5, min_periods=5).std()
    df['MA_5']               = df['Close'].rolling(window=5,  min_periods=5).mean()
    df['MA_10']              = df['Close'].rolling(window=10, min_periods=10).mean()

    # Hapus baris dengan NaN (akibat rolling & shift)
    df = df.dropna()

    # Pastikan hanya 11 kolom final yang tersimpan (tidak ada kolom intermediate)
    FINAL_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RETURN_LAG_1', 'RETURN_LAG_2',
        'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
        'MA_5', 'MA_10'
    ]
    df = df[FINAL_COLS]

    return df


# =============================================================================
# FUNGSI LOAD MODEL & DATA
# =============================================================================

def load_model_and_scaler():
    """
    Load model dan scaler yang sudah di-training.

    Returns:
        tuple: (model, scaler)

    Raises:
        FileNotFoundError: Jika file model atau scaler tidak ditemukan.
    """
    print("\n📦 Loading model dan scaler...")

    # Cek keberadaan file
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

    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model loaded  : {MODEL_PATH}")

    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✓ Scaler loaded : {SCALER_PATH}")

    return model, scaler


def load_and_prepare_data():
    """
    Load data historis dan lakukan feature engineering.

    Returns:
        pd.DataFrame: DataFrame dengan 11 kolom fitur final, siap prediksi.

    Raises:
        FileNotFoundError: Jika file data tidak ditemukan.
    """
    print("\n📊 Loading data historis...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data tidak ditemukan: {DATA_PATH}\n"
            f"Pastikan Anda sudah menjalankan training untuk generate data!"
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    invalid_index_mask = df.index.isna()
    if invalid_index_mask.any():
        dropped_rows = int(invalid_index_mask.sum())
        df = df.loc[~invalid_index_mask].copy()
        print(f"  ⚠ {dropped_rows} baris dihapus karena tanggal tidak valid (NaT)")

    df.index = pd.DatetimeIndex(df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Index DataFrame gagal dikonversi ke DatetimeIndex. "
            "Periksa format kolom tanggal pada CSV."
        )

    if len(df.index) > 0 and not isinstance(df.index[0], pd.Timestamp):
        raise TypeError(
            "Elemen index bukan pandas.Timestamp setelah konversi. "
            "Pastikan kolom tanggal terbaca dengan benar."
        )

    df = df.sort_index()

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_numeric_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_numeric_cols:
        raise ValueError(
            f"Kolom numerik wajib tidak ditemukan: {missing_numeric_cols}"
        )

    for col in numeric_cols:
        series = df[col]
        if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
            cleaned = (
                series.astype(str)
                .str.replace(',', '', regex=False)
                .str.replace(r'[^0-9eE+\-.]', '', regex=True)
            )
            df[col] = pd.to_numeric(cleaned, errors='coerce')
        else:
            df[col] = pd.to_numeric(series, errors='coerce')

    invalid_numeric_mask = df[numeric_cols].isna().any(axis=1)
    if invalid_numeric_mask.any():
        dropped_rows = int(invalid_numeric_mask.sum())
        df = df.loc[~invalid_numeric_mask].copy()
        print(f"  ⚠ {dropped_rows} baris dihapus karena nilai numerik tidak valid")

    df[numeric_cols] = df[numeric_cols].astype(float)

    non_float_cols = [
        col for col in numeric_cols if not pd.api.types.is_float_dtype(df[col])
    ]
    if non_float_cols:
        raise TypeError(
            f"Kolom berikut bukan float setelah konversi: {non_float_cols}. "
            f"dtypes saat ini: {df[numeric_cols].dtypes.to_dict()}"
        )

    print(f"  ✓ Data loaded: {len(df)} baris")
    print(f"  ✓ Periode    : {df.index[0].date()} hingga {df.index[-1].date()}")

    # Feature engineering — hasil: df_feat dengan 11 kolom final
    print("\n🔧 Melakukan feature engineering...")
    df_feat = create_features(df)
    print(f"  ✓ Features created : {len(df_feat)} baris (setelah dropna)")
    print(f"  ✓ Kolom df_feat    : {df_feat.shape[1]} ← tepat 11, tidak ada intermediate")

    # Validasi fitur
    missing_features = [f for f in FEATURE_COLS if f not in df_feat.columns]
    if missing_features:
        raise ValueError(f"Fitur tidak ditemukan: {missing_features}")

    return df_feat


# =============================================================================
# FUNGSI PREDIKSI UTAMA
# =============================================================================

def predict_next_trading_day(model, scaler, df_recent, verbose=True):
    """
    Prediksi harga saham untuk hari perdagangan berikutnya.

    Args:
        model: Trained Keras model (SimpleRNN).
        scaler: Fitted MinMaxScaler dari training.
        df_recent (pd.DataFrame): DataFrame dengan 11 kolom fitur final.
        verbose (bool): Print detail proses prediksi. Default: True.

    Returns:
        dict: Hasil prediksi dengan metadata:
            - predicted_price: float - harga prediksi (Rp)
            - prediction_date: datetime - tanggal prediksi
            - last_actual_date: datetime - tanggal data terakhir
            - last_actual_price: float - harga terakhir (Rp)
            - price_change: float - perubahan harga absolut (Rp)
            - price_change_pct: float - perubahan harga persentase (%)

    Raises:
        ValueError: Jika data tidak cukup atau validasi gagal.
    """
    if verbose:
        print("\n" + "="*70)
        print("PREDIKSI NEXT TRADING DAY")
        print("="*70)

    # -------------------------------------------------------------------------
    # STEP 1: VALIDASI DATA
    # -------------------------------------------------------------------------
    if len(df_recent) < WINDOW_SIZE:
        raise ValueError(
            f"Data tidak cukup. Butuh minimal {WINDOW_SIZE} baris, "
            f"tersedia: {len(df_recent)}"
        )

    if verbose:
        print(f"\n✓ Validasi data: OK ({len(df_recent)} baris tersedia)")

    # -------------------------------------------------------------------------
    # STEP 2: EKSTRAKSI WINDOW (10 DATA TERAKHIR)
    # -------------------------------------------------------------------------
    last_window = df_recent[FEATURE_COLS].iloc[-WINDOW_SIZE:].values

    expected_window_shape = (WINDOW_SIZE, len(FEATURE_COLS))
    if last_window.shape != expected_window_shape:
        raise ValueError(
            f"Shape input sebelum scaling tidak sesuai. "
            f"Expected {expected_window_shape}, got {last_window.shape}"
        )

    if verbose:
        print(f"\n📊 Data Input:")
        print(f"  Window size      : {WINDOW_SIZE} hari")
        print(f"  Jumlah fitur     : {len(FEATURE_COLS)}")
        print(f"  Tanggal terakhir : {df_recent.index[-1].date()}")
        print(f"  Harga terakhir   : Rp {df_recent[TARGET_COL].iloc[-1]:,.2f}")

    # -------------------------------------------------------------------------
    # STEP 3: NORMALISASI DENGAN SCALER LAMA (NO REFIT!)
    # -------------------------------------------------------------------------
    scaler_n_features = getattr(scaler, 'n_features_in_', None)
    if scaler_n_features is None:
        scaler_n_features = len(getattr(scaler, 'scale_', []))

    if scaler_n_features != len(FEATURE_COLS):
        raise ValueError(
            f"Jumlah fitur tidak cocok dengan scaler training. "
            f"Scaler expects {scaler_n_features}, "
            f"namun pipeline prediksi memberikan {len(FEATURE_COLS)} fitur."
        )

    last_window_scaled = scaler.transform(last_window)

    if verbose:
        print(f"\n🔧 Preprocessing:")
        print(f"  Normalisasi      : MinMaxScaler (dari training)")
        print(f"  Shape sebelum    : {last_window.shape}")
        print(f"  Shape sesudah    : {last_window_scaled.shape}")

    # -------------------------------------------------------------------------
    # STEP 4: RESHAPE KE FORMAT RNN (1, 10, JUMLAH_FITUR)
    # -------------------------------------------------------------------------
    X_pred = last_window_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))

    expected_rnn_shape = (1, WINDOW_SIZE, len(FEATURE_COLS))
    if X_pred.shape != expected_rnn_shape:
        raise ValueError(
            f"Shape input RNN tidak sesuai. "
            f"Expected {expected_rnn_shape}, got {X_pred.shape}"
        )

    if verbose:
        print(f"  Reshape untuk RNN: {X_pred.shape}")
        print(f"    ├─ Batch size  : {X_pred.shape[0]}")
        print(f"    ├─ Timesteps   : {X_pred.shape[1]}")
        print(f"    └─ Features    : {X_pred.shape[2]}")

    # -------------------------------------------------------------------------
    # STEP 5: MODEL INFERENCE (PREDIKSI)
    # -------------------------------------------------------------------------
    y_pred_scaled = model.predict(X_pred, verbose=0)
    y_pred_scaled = y_pred_scaled.flatten()[0]

    if verbose:
        print(f"\n🤖 Model Inference:")
        print(f"  Output (scaled)  : {y_pred_scaled:.6f}")

    # -------------------------------------------------------------------------
    # STEP 6: INVERSE TRANSFORM (HANYA UNTUK KOLOM TARGET CLOSE)
    # -------------------------------------------------------------------------
    target_idx = FEATURE_COLS.index(TARGET_COL)
    predicted_price = (
        (y_pred_scaled - scaler.min_[target_idx]) / scaler.scale_[target_idx]
    )

    if verbose:
        print(f"  Inverse transform: Rp {predicted_price:,.2f}")

    # -------------------------------------------------------------------------
    # STEP 7: AMBIL TANGGAL TERAKHIR DARI DATASET
    # -------------------------------------------------------------------------
    last_actual_date  = df_recent.index[-1]
    last_actual_price = df_recent[TARGET_COL].iloc[-1]

    # -------------------------------------------------------------------------
    # STEP 8: GUNAKAN get_next_trading_day() UNTUK TANGGAL PREDIKSI
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 9: OUTPUT HASIL PREDIKSI
    # -------------------------------------------------------------------------
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
        'predicted_price'   : float(predicted_price),
        'prediction_date'   : prediction_date,
        'last_actual_date'  : last_actual_date,
        'last_actual_price' : float(last_actual_price),
        'price_change'      : float(price_change),
        'price_change_pct'  : float(price_change_pct)
    }


# =============================================================================
# FUNGSI SAVE HASIL PREDIKSI
# =============================================================================

def save_prediction_result(result):
    """
    Simpan hasil prediksi ke file JSON.

    Args:
        result (dict): Hasil prediksi dari predict_next_trading_day().

    Returns:
        str: Path file output.
    """
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
        'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(PREDICTION_OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    return PREDICTION_OUTPUT_PATH


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function untuk menjalankan pipeline prediksi."""
    print("="*70)
    print("SISTEM PREDIKSI HARGA SAHAM TLKM")
    print("Next Trading Day Prediction")
    print("="*70)

    try:
        model, scaler = load_model_and_scaler()
        df_feat       = load_and_prepare_data()

        result = predict_next_trading_day(
            model=model,
            scaler=scaler,
            df_recent=df_feat,
            verbose=True
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
        print("   python tlkm_rnn_main.py")
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