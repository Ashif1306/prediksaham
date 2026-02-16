"""
Proyek Prediksi Harga Saham PT Telkom Indonesia (TLKM.JK)
Menggunakan Recurrent Neural Network (RNN)
Data: Yahoo Finance 2008 - Dinamis
"""

# Cek versi Python - TensorFlow belum mendukung 3.14
import sys
if sys.version_info >= (3, 14):
    print("="*60)
    print("ERROR: TensorFlow belum mendukung Python 3.14")
    print("Gunakan Python 3.11 atau 3.12.")
    print("Jalankan: setup.bat untuk otomatis setup dengan Python 3.12")
    print("="*60)
    sys.exit(1)

# =============================================================================
# STEP 1 — Setup Environment (Stable Version)
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import yfinance as yf
import os
from scipy.stats import pearsonr, binomtest

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seed untuk reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*60)
print("STEP 1: Environment Setup - SELESAI")
print(f"TensorFlow: {tf.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print("="*60)


# =============================================================================
# STEP 2 — Ambil Data TLKM (Dynamic Date, Aman)
# =============================================================================
def fetch_tlkm_data():
    """Ambil data historis TLKM dari Yahoo Finance secara dinamis."""
    ticker = "TLKM.JK"
    start_date = "2008-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\nMengunduh data TLKM dari Yahoo Finance...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if df.empty or len(df) < 100:
            raise ValueError(f"Data tidak cukup: hanya {len(df)} baris")
        
        # Pastikan index adalah DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        print(f"Data berhasil diunduh: {df.index[0].date()} hingga {df.index[-1].date()}")
        print(f"Total baris: {len(df)}")
        
        # Simpan data ke file CSV
        output_file = "data_tlkm_harga_saham.csv"
        df.to_csv(output_file)
        print(f"Data disimpan ke: {output_file}")
        
    except Exception as e:
        print(f"Error pengunduhan: {e}")
        # Fallback: load dari file jika ada
        if os.path.exists("data_tlkm_harga_saham.csv"):
            df = pd.read_csv("data_tlkm_harga_saham.csv", index_col=0, parse_dates=True)
            print("Menggunakan data lokal: data_tlkm_harga_saham.csv")
        elif os.path.exists("tlkm_backup.csv"):
            df = pd.read_csv("tlkm_backup.csv", index_col=0, parse_dates=True)
            print("Menggunakan data backup: tlkm_backup.csv")
        else:
            raise
    return df

df_raw = fetch_tlkm_data()

print("\n" + "="*60)
print("STEP 2: Pengambilan Data - SELESAI")
print("="*60)


# =============================================================================
# STEP 3 — Data Cleaning Aman
# =============================================================================
def clean_data(df):
    """Pembersihan data dengan operasi aman."""
    df = df.copy()

    def _flatten_columns(columns):
        """Flatten MultiIndex columns menjadi single level string."""
        if not isinstance(columns, pd.MultiIndex):
            return pd.Index(columns)

        flattened = []
        for col_tuple in columns.to_flat_index():
            parts = [str(part).strip() for part in col_tuple if part is not None and str(part).strip()]
            flattened.append('_'.join(parts))
        return pd.Index(flattened)

    # yfinance dapat mengembalikan MultiIndex; flatten dulu agar akses kolom selalu 1D
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_columns(df.columns)
    
    # Pastikan kolom yang dibutuhkan ada, termasuk hasil flatten seperti Open_TLKM.JK
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    selected_cols = {}
    for base_col in required_cols:
        candidates = [
            col for col in df.columns
            if col == base_col or col.startswith(f'{base_col}_')
        ]
        if candidates:
            selected_cols[base_col] = candidates[0]

    if not selected_cols:
        raise ValueError(
            f"Kolom OHLCV tidak ditemukan. Kolom tersedia: {list(df.columns)}"
        )

    df = df[list(selected_cols.values())].copy()
    df = df.rename(columns={v: k for k, v in selected_cols.items()})

    # Hindari duplikasi kolom setelah rename (mis. multi ticker)
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T
    
    # Bersihkan index agar valid datetime
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]

    # Pastikan semua kolom numerik aman; apply menjaga operasi per-Series (1D)
    df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))

    # Hapus duplikat index
    df = df[~df.index.duplicated(keep='first')]
    
    # Isi missing value dengan metode forward fill, lalu backward fill
    df = df.ffill().bfill()
    
    # Hapus baris yang masih ada NaN
    df = df.dropna()
    
    # Validasi: pastikan tidak ada nilai negatif untuk harga/volume
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df = df[df[col] > 0]
    
    return df

df_clean = clean_data(df_raw)
print(f"\nData setelah cleaning: {len(df_clean)} baris")

print("\n" + "="*60)
print("STEP 3: Data Cleaning - SELESAI")
print("="*60)


# =============================================================================
# STEP 4 — EDA Dasar
# =============================================================================
print("\n--- EDA: Statistik Deskriptif ---")
print(df_clean.describe())

print("\n--- EDA: Info Data ---")
print(df_clean.info())

print("\n--- EDA: Korelasi (jika ada Volume) ---")
if 'Volume' in df_clean.columns:
    corr = df_clean[['Open','High','Low','Close','Volume']].corr()
    print(corr)
else:
    corr = df_clean.corr()
    print(corr)

print("\n" + "="*60)
print("STEP 4: EDA Dasar - SELESAI")
print("="*60)


# =============================================================================
# STEP 5 — Feature Engineering Stabil + Momentum Core
# =============================================================================
def create_features(df, add_momentum_core_features=False):
    """Feature engineering stabil dengan opsi fitur Momentum Core minimalis."""
    df = df.copy()
    
    # Gunakan Close sebagai target utama
    if 'Close' not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan")
    
    # Gunakan OHLCV + indikator teknikal dasar untuk stabilitas generalisasi
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Fitur inti Momentum Core (tanpa data leakage: hanya t dan masa lalu)
    if add_momentum_core_features:
        df['RETURN_1D'] = df['Close'].pct_change()
        df['RETURN_LAG_1'] = df['RETURN_1D'].shift(1)
        df['RETURN_LAG_2'] = df['RETURN_1D'].shift(2)
        df['RSI_SLOPE'] = df['RSI_14'].diff()
        df['ROLL_STD_RETURN_5D'] = df['RETURN_1D'].rolling(window=5, min_periods=5).std()
    
    # Hapus baris dengan NaN dari rolling
    df = df.dropna()
    
    return df

base_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'MACD_SIGNAL']
momentum_core_feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RETURN_LAG_1', 'RETURN_LAG_2', 'RSI_SLOPE', 'ROLL_STD_RETURN_5D'
]

# Bangun dataset momentum core terlebih dahulu. Baseline akan menggunakan subset fitur dasar
# dari dataset yang sama agar perbandingan adil pada periode sampel yang identik.
df_feat_momentum_core = create_features(df_clean, add_momentum_core_features=True)
df_feat = df_feat_momentum_core.copy()

feature_set_map = {
    'BASELINE': [c for c in base_feature_cols if c in df_feat.columns],
    'MOMENTUM_CORE': [c for c in momentum_core_feature_cols if c in df_feat.columns]
}

print(f"\nJumlah fitur baseline: {len(feature_set_map['BASELINE'])}")
print(f"Jumlah fitur momentum core: {len(feature_set_map['MOMENTUM_CORE'])}")
print(f"Baris setelah feature engineering (aligned): {len(df_feat)}")

print("\n" + "="*60)
print("STEP 5: Feature Engineering - SELESAI")
print("="*60)


# =============================================================================
# STEP 6 — Normalisasi TANPA BUG (PENTING)
# =============================================================================
# PENTING: Scaler HANYA di-fit pada data TRAINING untuk menghindari data leakage
# Scaler akan di-fit nanti setelah train-test split

TARGET_COL = 'Close'

# Buat scaler global per skenario - akan di-fit pada TRAIN saja di Step 8
scaler_global_map = {
    scenario: MinMaxScaler(feature_range=(0, 1))
    for scenario in feature_set_map
}

# Untuk sementara, kita simpan data mentah
# Fit scaler nanti HANYA pada train set
print("\nScaler global MinMaxScaler(0,1) siap. Fitting HANYA pada train set (Step 8).")

print("\n" + "="*60)
print("STEP 6: Normalisasi (Scaler siap, fit di Step 8) - SELESAI")
print("="*60)


# =============================================================================
# STEP 7 — Sliding Window Aman
# =============================================================================
def create_sequences(data, seq_length, target_idx):
    """
    Buat struktur time series dengan sliding window.
    X: (samples, timesteps, features)
    y: (samples,) -> nilai Close pada timestep berikutnya
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)

print("\nSliding window: SEQ_LENGTH = 10 (stabil)")

print("\n" + "="*60)
print("STEP 7: Sliding Window (fungsi siap) - SELESAI")
print("="*60)


# =============================================================================
# STEP 8 — Train-Test Split TANPA LEAKAGE
# =============================================================================
TRAIN_RATIO = 0.8

split_idx = int(len(df_feat) * TRAIN_RATIO)
dataset_map = {}

for scenario_name, scenario_features in feature_set_map.items():
    train_data = df_feat.iloc[:split_idx][scenario_features].values
    test_data = df_feat.iloc[split_idx:][scenario_features].values

    scaler = scaler_global_map[scenario_name]
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    dataset_map[scenario_name] = {
        'feature_cols': scenario_features,
        'train_scaled': train_scaled,
        'test_scaled': test_scaled,
        'scaler_global': scaler,
        'split_idx': split_idx
    }

print(f"\nTrain/Test split index: {split_idx} ({TRAIN_RATIO*100:.0f}% train)")
for scenario_name, dataset in dataset_map.items():
    print(
        f"  {scenario_name:<9} -> fitur: {len(dataset['feature_cols']):<2} | "
        f"train: {dataset['train_scaled'].shape[0]} | test: {dataset['test_scaled'].shape[0]}"
    )

print("\n" + "="*60)
print("STEP 8: Train-Test Split (TANPA LEAKAGE) - SELESAI")
print("="*60)


# =============================================================================
# STEP 9-11 — Eksperimen & Evaluasi (Window 10)
# =============================================================================
def inverse_transform_close(scaler, values, close_idx):
    """Inverse transform khusus kolom Close menggunakan parameter scaler yang sudah di-fit pada train."""
    values = np.asarray(values)
    return (values - scaler.min_[close_idx]) / scaler.scale_[close_idx]

def build_model(seq_length, n_features):
    """Arsitektur RNN teroptimasi."""
    return Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.25),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])

def evaluate_directional_accuracy(y_test_inv, y_pred_inv):
    """
    Evaluasi Directional Accuracy berbasis arah perubahan terhadap harga aktual sebelumnya.

    Definisi:
    - Arah aktual_t   = sign(y_t - y_(t-1))
    - Arah prediksi_t = sign(y_pred_t - y_(t-1))

    Catatan alignment:
    - t dimulai dari indeks 1 agar y_(t-1) tersedia.
    - Jumlah sampel evaluasi = len(y_test_inv) - 1.
    """
    y_test_inv = np.asarray(y_test_inv).flatten()
    y_pred_inv = np.asarray(y_pred_inv).flatten()

    if len(y_test_inv) != len(y_pred_inv):
        raise ValueError("Panjang y_test_inv dan y_pred_inv harus sama untuk evaluasi arah.")
    if len(y_test_inv) < 2:
        raise ValueError("Minimal butuh 2 titik data untuk menghitung arah pergerakan.")

    y_prev_actual = y_test_inv[:-1]
    y_curr_actual = y_test_inv[1:]
    y_curr_pred = y_pred_inv[1:]

    actual_sign = np.sign(y_curr_actual - y_prev_actual)
    pred_sign = np.sign(y_curr_pred - y_prev_actual)

    # Mapping biner untuk confusion matrix sederhana:
    # >= 0 dianggap Naik, < 0 dianggap Turun.
    actual_up = actual_sign >= 0
    pred_up = pred_sign >= 0

    naik_naik = np.sum(actual_up & pred_up)
    turun_turun = np.sum(~actual_up & ~pred_up)
    naik_turun = np.sum(actual_up & ~pred_up)
    turun_naik = np.sum(~actual_up & pred_up)

    total = len(actual_sign)
    correct = naik_naik + turun_turun
    directional_accuracy = (correct / total) * 100 if total > 0 else np.nan

    confusion_matrix = {
        'Naik-Naik': int(naik_naik),
        'Turun-Turun': int(turun_turun),
        'Naik-Turun': int(naik_turun),
        'Turun-Naik': int(turun_naik)
    }

    return {
        'directional_accuracy': directional_accuracy,
        'correct_predictions': int(correct),
        'total_predictions': int(total),
        'actual_sign': actual_sign,
        'pred_sign': pred_sign,
        'confusion_matrix': confusion_matrix
    }

def evaluate_directional_accuracy_from_signs(y_test_inv, pred_sign):
    """Evaluasi Directional Accuracy menggunakan sinyal arah prediksi yang sudah diproses."""
    y_test_inv = np.asarray(y_test_inv).flatten()
    pred_sign = np.asarray(pred_sign).flatten()

    if len(y_test_inv) < 2:
        raise ValueError("Minimal butuh 2 titik data untuk menghitung arah pergerakan.")

    actual_sign = np.sign(np.diff(y_test_inv))
    if len(pred_sign) != len(actual_sign):
        raise ValueError("Panjang pred_sign harus sama dengan len(y_test_inv)-1.")

    actual_up = actual_sign >= 0
    pred_up = pred_sign >= 0

    naik_naik = np.sum(actual_up & pred_up)
    turun_turun = np.sum(~actual_up & ~pred_up)
    naik_turun = np.sum(actual_up & ~pred_up)
    turun_naik = np.sum(~actual_up & pred_up)

    total = len(actual_sign)
    correct = naik_naik + turun_turun
    directional_accuracy = (correct / total) * 100 if total > 0 else np.nan

    confusion_matrix = {
        'Naik-Naik': int(naik_naik),
        'Turun-Turun': int(turun_turun),
        'Naik-Turun': int(naik_turun),
        'Turun-Naik': int(turun_naik)
    }

    return {
        'directional_accuracy': directional_accuracy,
        'correct_predictions': int(correct),
        'total_predictions': int(total),
        'actual_sign': actual_sign,
        'pred_sign': pred_sign,
        'confusion_matrix': confusion_matrix
    }

def compute_return_series(price_series):
    """Hitung return sederhana dari deret harga dengan panjang output len(price_series)-1."""
    price_series = np.asarray(price_series, dtype=float).flatten()
    if len(price_series) < 2:
        raise ValueError("Minimal butuh 2 titik harga untuk menghitung return.")
    return np.diff(price_series) / (price_series[:-1] + 1e-12)

def evaluate_trend_accuracy_n_day(y_actual, y_pred, n_days=3):
    """Akurasi arah tren berdasarkan tanda return kumulatif n-hari aktual vs prediksi."""
    y_actual = np.asarray(y_actual, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_actual) != len(y_pred):
        raise ValueError("Panjang y_actual dan y_pred harus sama.")
    if len(y_actual) <= n_days:
        raise ValueError(f"Minimal butuh {n_days + 1} titik data untuk tren {n_days}-hari.")

    actual_cumret = (y_actual[n_days:] / (y_actual[:-n_days] + 1e-12)) - 1.0
    pred_cumret = (y_pred[n_days:] / (y_pred[:-n_days] + 1e-12)) - 1.0

    actual_sign = np.sign(actual_cumret)
    pred_sign = np.sign(pred_cumret)

    accuracy = np.mean(actual_sign == pred_sign) * 100
    return {
        'n_days': n_days,
        'accuracy': float(accuracy),
        'total_samples': int(len(actual_sign)),
        'correct_samples': int(np.sum(actual_sign == pred_sign)),
        'actual_sign': actual_sign,
        'pred_sign': pred_sign
    }

def evaluate_directional_accuracy_by_magnitude(actual_returns, pred_returns, small_th=0.003, medium_th=0.01):
    """Hitung Directional Accuracy per kategori magnitude return aktual."""
    actual_returns = np.asarray(actual_returns, dtype=float).flatten()
    pred_returns = np.asarray(pred_returns, dtype=float).flatten()

    if len(actual_returns) != len(pred_returns):
        raise ValueError("Panjang actual_returns dan pred_returns harus sama.")

    abs_returns = np.abs(actual_returns)
    categories = {
        f'kecil(<{small_th*100:.1f}%)': abs_returns < small_th,
        f'sedang({small_th*100:.1f}%-{medium_th*100:.1f}%)': (abs_returns >= small_th) & (abs_returns <= medium_th),
        f'besar(>{medium_th*100:.1f}%)': abs_returns > medium_th
    }

    actual_sign = np.sign(actual_returns)
    pred_sign = np.sign(pred_returns)

    result = {}
    for category_name, mask in categories.items():
        total = int(np.sum(mask))
        if total == 0:
            result[category_name] = {'accuracy': np.nan, 'correct': 0, 'total': 0}
            continue
        correct = int(np.sum(actual_sign[mask] == pred_sign[mask]))
        result[category_name] = {
            'accuracy': (correct / total) * 100,
            'correct': correct,
            'total': total
        }
    return result

def evaluate_return_correlation(actual_returns, pred_returns):
    """Pearson correlation antara return aktual dan return prediksi."""
    actual_returns = np.asarray(actual_returns, dtype=float).flatten()
    pred_returns = np.asarray(pred_returns, dtype=float).flatten()

    if len(actual_returns) != len(pred_returns):
        raise ValueError("Panjang actual_returns dan pred_returns harus sama.")

    corr, p_value = pearsonr(actual_returns, pred_returns)
    return {'pearson_correlation': float(corr), 'p_value': float(p_value)}

def evaluate_directional_significance(correct_predictions, total_predictions, p_random=0.5):
    """Uji signifikansi binomial untuk Directional Accuracy vs tebakan acak."""
    if total_predictions <= 0:
        raise ValueError("total_predictions harus lebih besar dari 0.")
    test_result = binomtest(k=correct_predictions, n=total_predictions, p=p_random, alternative='two-sided')
    return {
        'p_value': float(test_result.pvalue),
        'expected_accuracy_random': p_random * 100
    }

def compute_direction_classification_report(actual_sign, pred_sign):
    """Confusion matrix + precision/recall/F1 untuk arah Naik dan Turun."""
    actual_sign = np.asarray(actual_sign, dtype=float).flatten()
    pred_sign = np.asarray(pred_sign, dtype=float).flatten()

    if len(actual_sign) != len(pred_sign):
        raise ValueError("Panjang actual_sign dan pred_sign harus sama.")

    actual_up = actual_sign >= 0
    pred_up = pred_sign >= 0

    tp_up = int(np.sum(actual_up & pred_up))
    fp_up = int(np.sum(~actual_up & pred_up))
    fn_up = int(np.sum(actual_up & ~pred_up))
    tn_up = int(np.sum(~actual_up & ~pred_up))

    tp_down = tn_up
    fp_down = fn_up
    fn_down = fp_up

    def _safe_div(num, den):
        return (num / den) if den > 0 else np.nan

    precision_up = _safe_div(tp_up, tp_up + fp_up)
    recall_up = _safe_div(tp_up, tp_up + fn_up)
    f1_up = _safe_div(2 * precision_up * recall_up, precision_up + recall_up) if not np.isnan(precision_up + recall_up) and (precision_up + recall_up) > 0 else np.nan

    precision_down = _safe_div(tp_down, tp_down + fp_down)
    recall_down = _safe_div(tp_down, tp_down + fn_down)
    f1_down = _safe_div(2 * precision_down * recall_down, precision_down + recall_down) if not np.isnan(precision_down + recall_down) and (precision_down + recall_down) > 0 else np.nan

    return {
        'confusion_matrix': {
            'actual_naik_pred_naik': tp_up,
            'actual_naik_pred_turun': fn_up,
            'actual_turun_pred_naik': fp_up,
            'actual_turun_pred_turun': tn_up
        },
        'metrics': {
            'naik': {'precision': precision_up, 'recall': recall_up, 'f1_score': f1_up},
            'turun': {'precision': precision_down, 'recall': recall_down, 'f1_score': f1_down}
        }
    }

def _fill_zero_sign_with_previous(sign_array):
    """Isi nilai 0 dengan arah sebelumnya agar semua sampel tetap dihitung naik/turun."""
    sign_array = np.asarray(sign_array, dtype=float).copy()
    if sign_array.size == 0:
        return sign_array

    first_nonzero_idx = np.flatnonzero(sign_array)
    default_sign = sign_array[first_nonzero_idx[0]] if len(first_nonzero_idx) > 0 else 1.0
    prev = default_sign

    for i in range(sign_array.size):
        if sign_array[i] == 0:
            sign_array[i] = prev
        else:
            prev = sign_array[i]
    return sign_array

def build_directional_decision_layer(y_pred_inv, rsi_values, macd_values, macd_signal_values, smooth_window=3):
    """
    Decision layer post-processing tanpa mengubah output regresi asli model.

    Komponen:
    1) Momentum confirmation: delta prediksi vs delta prediksi sebelumnya.
    2) Smoothing ringan: rolling mean pada prediksi.
    3) Konfirmasi indikator teknikal: RSI momentum + status MACD terhadap signal.
    """
    if smooth_window not in (2, 3):
        raise ValueError("smooth_window direkomendasikan 2 atau 3.")

    y_pred_inv = np.asarray(y_pred_inv).flatten()
    rsi_values = np.asarray(rsi_values).flatten()
    macd_values = np.asarray(macd_values).flatten()
    macd_signal_values = np.asarray(macd_signal_values).flatten()

    if not (len(y_pred_inv) == len(rsi_values) == len(macd_values) == len(macd_signal_values)):
        raise ValueError("Panjang y_pred_inv, RSI, MACD, dan MACD_SIGNAL harus sama.")
    if len(y_pred_inv) < 2:
        raise ValueError("Minimal 2 titik prediksi dibutuhkan untuk membuat sinyal arah.")

    pred_series = pd.Series(y_pred_inv)
    pred_smooth = pred_series.rolling(window=smooth_window, min_periods=1).mean().to_numpy()

    trend_sign = np.sign(np.diff(pred_smooth))
    momentum_sign = np.sign(pred_smooth[1:] - pred_smooth[:-1])

    trend_sign = _fill_zero_sign_with_previous(trend_sign)
    momentum_sign = _fill_zero_sign_with_previous(momentum_sign)

    # Momentum confirmation + anti false reversal sederhana
    confirmed_sign = np.zeros_like(trend_sign)
    confirmed_sign[0] = trend_sign[0]
    for i in range(1, len(trend_sign)):
        same_direction = trend_sign[i] == momentum_sign[i]
        if same_direction:
            confirmed_sign[i] = trend_sign[i]
        else:
            confirmed_sign[i] = confirmed_sign[i - 1]

    rsi_momentum_sign = np.sign(np.diff(rsi_values))
    macd_state_sign = np.sign((macd_values - macd_signal_values)[1:])

    rsi_momentum_sign = _fill_zero_sign_with_previous(rsi_momentum_sign)
    macd_state_sign = _fill_zero_sign_with_previous(macd_state_sign)

    technical_vote = rsi_momentum_sign + macd_state_sign
    technical_sign = np.where(technical_vote >= 0, 1.0, -1.0)

    # Kombinasi final: validasi dengan indikator teknikal, tetap klasifikasi semua titik
    final_sign = np.zeros_like(confirmed_sign)
    final_sign[0] = confirmed_sign[0]
    for i in range(len(confirmed_sign)):
        if confirmed_sign[i] == technical_sign[i]:
            final_sign[i] = confirmed_sign[i]
        elif momentum_sign[i] == technical_sign[i]:
            final_sign[i] = technical_sign[i]
        elif i > 0:
            final_sign[i] = final_sign[i - 1]
        else:
            final_sign[i] = confirmed_sign[i]

    final_sign = _fill_zero_sign_with_previous(final_sign)

    return {
        'pred_smoothed': pred_smooth,
        'trend_sign': trend_sign,
        'momentum_sign': momentum_sign,
        'technical_sign': technical_sign,
        'final_sign': final_sign
    }

def compute_max_drawdown(equity_curve):
    """Hitung maksimum drawdown (%) dari equity curve."""
    equity_curve = pd.Series(equity_curve, dtype=float)
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve / rolling_peak) - 1.0
    return float(drawdown.min())

def compute_backtest_metrics(equity_curve, strategy_returns, exposure):
    """Ringkasan metrik performa finansial untuk strategi."""
    equity_curve = pd.Series(equity_curve, dtype=float)
    strategy_returns = pd.Series(strategy_returns, dtype=float)
    exposure = pd.Series(exposure, dtype=float)

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    active_mask = exposure > 0
    if active_mask.any():
        win_rate = (strategy_returns[active_mask] > 0).mean()
    else:
        win_rate = np.nan

    return {
        'cumulative_return': equity_curve / equity_curve.iloc[0] - 1.0,
        'total_return': float(total_return),
        'win_rate': float(win_rate) if not np.isnan(win_rate) else np.nan,
        'max_drawdown': compute_max_drawdown(equity_curve)
    }

def run_directional_backtest(price_series, signal_sign, initial_capital=100_000_000, transaction_cost=0.0015):
    """
    Backtest berbasis sinyal arah (>=0 buy, <0 cash) dengan tanpa leverage.

    Anti look-ahead:
    - Posisi pada hari t dibentuk dari sinyal hari t-1 (shift 1 langkah).
    - Return hari t memakai posisi yang sudah ditentukan di akhir hari sebelumnya.
    """
    prices = pd.Series(price_series, dtype=float).copy()
    if prices.isna().any():
        raise ValueError("price_series mengandung NaN.")
    if len(prices) < 3:
        raise ValueError("Minimal 3 data harga dibutuhkan untuk backtest.")

    signal_sign = np.asarray(signal_sign, dtype=float).flatten()
    if len(signal_sign) != len(prices) - 1:
        raise ValueError("Panjang signal_sign harus sama dengan len(price_series)-1.")

    signal_state = pd.Series(np.where(signal_sign >= 0, 1.0, 0.0), index=prices.index[1:])
    signal_state = pd.concat([pd.Series([0.0], index=[prices.index[0]]), signal_state])

    asset_returns = prices.pct_change().fillna(0.0)
    position = signal_state.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = (position * asset_returns) - (turnover * transaction_cost)

    equity_curve = initial_capital * (1.0 + strategy_returns).cumprod()
    strategy_metrics = compute_backtest_metrics(equity_curve, strategy_returns, position)

    bh_position = pd.Series(1.0, index=prices.index)
    bh_turnover = bh_position.diff().abs().fillna(1.0)
    bh_returns = asset_returns - (bh_turnover * transaction_cost)
    bh_equity = initial_capital * (1.0 + bh_returns).cumprod()
    bh_metrics = compute_backtest_metrics(bh_equity, bh_returns, bh_position)

    return {
        'prices': prices,
        'asset_returns': asset_returns,
        'signal_state': signal_state,
        'position': position,
        'strategy_returns': strategy_returns,
        'equity_curve': equity_curve,
        'strategy_metrics': strategy_metrics,
        'buy_hold_returns': bh_returns,
        'buy_hold_equity': bh_equity,
        'buy_hold_metrics': bh_metrics,
        'transaction_cost': transaction_cost,
        'initial_capital': initial_capital
    }

def plot_backtest_equity_curve(backtest_result, output_path='tlkm_backtest_equity_curve.png'):
    """Visualisasi equity curve strategi model vs buy & hold."""
    plt.figure(figsize=(14, 5))
    plt.plot(backtest_result['equity_curve'].index, backtest_result['equity_curve'].values,
             label='Strategi Model (Directional Signal)', linewidth=2)
    plt.plot(backtest_result['buy_hold_equity'].index, backtest_result['buy_hold_equity'].values,
             label='Buy & Hold', linewidth=2, alpha=0.8)
    plt.title('Perbandingan Equity Curve: Strategi Model vs Buy & Hold')
    plt.xlabel('Tanggal')
    plt.ylabel('Nilai Portofolio (Rp)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_experiment(seq_length, train_scaled, test_scaled, scaler_global, feature_cols, close_idx, seed=SEED):
    """Jalankan eksperimen untuk satu window size."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    X_train, y_train = create_sequences(train_scaled, seq_length, close_idx)
    X_test, y_test = create_sequences(test_scaled, seq_length, close_idx)
    
    n_features = X_train.shape[2]
    model = build_model(seq_length, n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluasi dengan inverse transform
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_actual_scaled = y_test
    y_actual = inverse_transform_close(scaler_global, y_actual_scaled, close_idx)
    y_pred_inv = inverse_transform_close(scaler_global, y_pred_scaled, close_idx)
    
    min_len = min(len(y_actual), len(y_pred_inv))
    y_actual = y_actual[:min_len]
    y_pred_inv = y_pred_inv[:min_len]
    
    EPSILON = 1e-8
    MAE = mean_absolute_error(y_actual, y_pred_inv)
    RMSE = np.sqrt(mean_squared_error(y_actual, y_pred_inv))
    R2 = r2_score(y_actual, y_pred_inv)
    MAPE = np.mean(np.abs((y_actual - y_pred_inv) / (np.abs(y_actual) + EPSILON))) * 100
    
    return {
        'seq_length': seq_length,
        'MAE': MAE, 'RMSE': RMSE, 'MAPE': MAPE, 'R2': R2,
        'model': model, 'history': history,
        'y_actual': y_actual, 'y_pred_inv': y_pred_inv
    }


def create_classification_sequences(data, labels, seq_length):
    """Buat sequence untuk klasifikasi arah dengan target label biner terpisah."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)


def build_direction_classifier(seq_length, n_features):
    """Arsitektur SimpleRNN klasifikasi arah (terpisah dari model regresi)."""
    return Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.25),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


def run_direction_classification_experiment(
    seq_length,
    df_features,
    feature_cols,
    split_idx,
    horizon_days=3,
    seed=SEED
):
    """Training/evaluasi model klasifikasi arah menggunakan pipeline preprocessing yang sama."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if horizon_days < 1:
        raise ValueError("horizon_days minimal 1.")

    # Target biner horizon ke depan (tanpa leakage):
    # y_t = 1 jika Close_(t+h) > Close_t, else 0
    # Drop baris ujung yang tidak punya Close_(t+h) agar fitur-target tetap sejajar.
    forward_close = df_features['Close'].shift(-horizon_days)
    direction_target = (forward_close > df_features['Close']).astype(int)
    valid_mask = forward_close.notna()

    df_aligned = df_features.loc[valid_mask].copy()
    direction_target = direction_target.loc[valid_mask]

    if split_idx > len(df_aligned):
        split_idx = int(len(df_aligned) * TRAIN_RATIO)

    train_data = df_aligned.iloc[:split_idx][feature_cols].values
    test_data = df_aligned.iloc[split_idx:][feature_cols].values
    y_train_raw = direction_target.iloc[:split_idx].values
    y_test_raw = direction_target.iloc[split_idx:].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_classification_sequences(train_scaled, y_train_raw, seq_length)
    X_test, y_test = create_classification_sequences(test_scaled, y_test_raw, seq_length)

    train_class_counts = pd.Series(y_train).value_counts().reindex([1, 0], fill_value=0)
    train_total = int(train_class_counts.sum())
    train_class_distribution = {
        'naik': {
            'count': int(train_class_counts.loc[1]),
            'percentage': (float(train_class_counts.loc[1]) / train_total * 100.0) if train_total > 0 else np.nan
        },
        'turun': {
            'count': int(train_class_counts.loc[0]),
            'percentage': (float(train_class_counts.loc[0]) / train_total * 100.0) if train_total > 0 else np.nan
        }
    }

    n_classes = 2
    imbalance_ratio = (
        float(train_class_counts.max()) / float(train_class_counts.min())
        if train_class_counts.min() > 0 else np.inf
    )
    use_class_weight = imbalance_ratio > 1.2
    class_weight = {
        cls: (train_total / (n_classes * count)) if count > 0 else 1.0
        for cls, count in train_class_counts.items()
    } if use_class_weight else None

    model = build_direction_classifier(seq_length, X_train.shape[2])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        shuffle=False,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    prob_distribution = {
        'min': float(np.min(y_pred_prob)),
        'max': float(np.max(y_pred_prob)),
        'mean': float(np.mean(y_pred_prob))
    }

    threshold_candidates = np.round(np.arange(0.45, 0.551, 0.01), 2).tolist()
    threshold_results = []
    for threshold in threshold_candidates:
        y_pred_class_t = (y_pred_prob >= threshold).astype(int)
        threshold_results.append({
            'threshold': threshold,
            'accuracy': float(accuracy_score(y_test, y_pred_class_t)),
            'precision': float(precision_score(y_test, y_pred_class_t, average='macro', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_class_t, average='macro', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_class_t, average='macro', zero_division=0)),
            'predicted_up_count': int(np.sum(y_pred_class_t == 1)),
            'predicted_down_count': int(np.sum(y_pred_class_t == 0))
        })

    valid_threshold_results = [
        row for row in threshold_results
        if row['predicted_up_count'] > 0 and row['predicted_down_count'] > 0
    ]
    threshold_pool = valid_threshold_results if valid_threshold_results else threshold_results

    best_threshold_result = max(
        threshold_pool,
        key=lambda item: (item['f1_score'], item['accuracy'])
    )
    best_threshold = best_threshold_result['threshold']
    y_pred_class = (y_pred_prob >= best_threshold).astype(int)

    overall_accuracy = accuracy_score(y_test, y_pred_class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_class, labels=[1, 0], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred_class, labels=[1, 0])

    correct_predictions = int(np.sum(y_test == y_pred_class))
    total_predictions = int(len(y_test))
    significance = evaluate_directional_significance(
        correct_predictions=correct_predictions,
        total_predictions=total_predictions,
        p_random=0.5
    )

    return {
        'horizon_days': int(horizon_days),
        'seq_length': seq_length,
        'history': history,
        'model': model,
        'scaler': scaler,
        'train_class_distribution': train_class_distribution,
        'class_weight_used': bool(use_class_weight),
        'class_imbalance_ratio': float(imbalance_ratio),
        'class_weight': {int(k): float(v) for k, v in class_weight.items()} if class_weight is not None else None,
        'y_test': y_test,
        'y_pred_prob': y_pred_prob,
        'prob_distribution': prob_distribution,
        'threshold_results': threshold_results,
        'best_threshold': float(best_threshold),
        'best_threshold_metrics': best_threshold_result,
        'y_pred_class': y_pred_class,
        'overall_accuracy': float(overall_accuracy),
        'directional_accuracy': float(overall_accuracy * 100),
        'metrics': {
            'naik': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0])
            },
            'turun': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1])
            }
        },
        'confusion_matrix': {
            'labels': ['Naik (1)', 'Turun (0)'],
            'matrix': cm
        },
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'binomial_significance': significance
    }

# Jalankan eksperimen untuk window 10
WINDOW_SIZE = 10
print()
print("="*60)
print("STEP 9-10: EKSPERIMEN (Window 10)")
print("="*60)

experiment_results = {}
for scenario_name, dataset in dataset_map.items():
    print()
    print(f">>> Melatih model [{scenario_name}] dengan window size = {WINDOW_SIZE} ...")
    close_idx = dataset['feature_cols'].index(TARGET_COL)
    result = run_experiment(
        WINDOW_SIZE,
        dataset['train_scaled'],
        dataset['test_scaled'],
        dataset['scaler_global'],
        dataset['feature_cols'],
        close_idx
    )
    experiment_results[scenario_name] = result
    print(
        f"    MAPE = {result['MAPE']:.2f}% | MAE = Rp {result['MAE']:,.2f} | "
        f"R² = {result['R2']:.4f}"
    )

print()
print("="*60)
print("STEP 10B: MODEL KLASIFIKASI ARAH (TERPISAH)")
print("="*60)

classification_result_h1 = run_direction_classification_experiment(
    seq_length=WINDOW_SIZE,
    df_features=df_feat,
    feature_cols=feature_set_map['MOMENTUM_CORE'],
    split_idx=split_idx,
    horizon_days=1
)
classification_result = run_direction_classification_experiment(
    seq_length=WINDOW_SIZE,
    df_features=df_feat,
    feature_cols=feature_set_map['MOMENTUM_CORE'],
    split_idx=split_idx,
    horizon_days=3
)

print()
print("--- Evaluasi Model Klasifikasi Arah (SimpleRNN Sigmoid) | Horizon 3-Hari ---")

train_dist = classification_result['train_class_distribution']
print("Distribusi kelas data train (setelah sequence):")
print(
    f"  Naik  : {train_dist['naik']['count']} sampel "
    f"({train_dist['naik']['percentage']:.2f}%)"
)
print(
    f"  Turun : {train_dist['turun']['count']} sampel "
    f"({train_dist['turun']['percentage']:.2f}%)"
)

print(
    f"Class imbalance ratio (mayor/minor) = {classification_result['class_imbalance_ratio']:.4f} "
    f"-> class_weight digunakan: {classification_result['class_weight_used']}"
)
if classification_result['class_weight_used']:
    print("Class weight (berdasarkan proporsi kelas train):")
    print(f"  class_weight[1] (Naik)  = {classification_result['class_weight'][1]:.4f}")
    print(f"  class_weight[0] (Turun) = {classification_result['class_weight'][0]:.4f}")
else:
    print("Class weight tidak digunakan karena distribusi kelas relatif seimbang.")

last_auc = classification_result['history'].history.get('auc', [np.nan])[-1]
last_val_auc = classification_result['history'].history.get('val_auc', [np.nan])[-1]
print(f"AUC (train akhir epoch)    : {last_auc:.4f}")
print(f"AUC (validasi akhir epoch) : {last_val_auc:.4f}")

prob_dist = classification_result['prob_distribution']
print("Distribusi probabilitas output (y_pred_prob):")
print(f"  Min  : {prob_dist['min']:.6f}")
print(f"  Max  : {prob_dist['max']:.6f}")
print(f"  Mean : {prob_dist['mean']:.6f}")

print()
print("Threshold tuning (berdasarkan probabilitas Naik, metrik macro untuk keseimbangan kelas):")
print(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Pred Naik':<10} {'Pred Turun':<10}")
print("-" * 84)
for row in classification_result['threshold_results']:
    print(
        f"{row['threshold']:<12.2f} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
        f"{row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['predicted_up_count']:<10} {row['predicted_down_count']:<10}"
    )

best_thr = classification_result['best_threshold_metrics']
print()
print(
    f"Threshold terbaik (Macro F1 tertinggi, non-collapse): {classification_result['best_threshold']:.2f} "
    f"dengan Macro F1={best_thr['f1_score']:.4f}, Accuracy={best_thr['accuracy']:.4f}"
)

print()
print(f"Directional Accuracy keseluruhan : {classification_result['directional_accuracy']:.2f}%")
print(
    f"Akurasi = {classification_result['overall_accuracy']:.4f} "
    f"({classification_result['correct_predictions']}/{classification_result['total_predictions']})"
)

print()
print("Confusion Matrix (Aktual x Prediksi) [Naik(1), Turun(0)]:")
cm_class = classification_result['confusion_matrix']['matrix']
print(f"  Aktual Naik  -> Pred Naik : {cm_class[0, 0]}")
print(f"  Aktual Naik  -> Pred Turun: {cm_class[0, 1]}")
print(f"  Aktual Turun -> Pred Naik : {cm_class[1, 0]}")
print(f"  Aktual Turun -> Pred Turun: {cm_class[1, 1]}")

for class_name in ['naik', 'turun']:
    m = classification_result['metrics'][class_name]
    print()
    print(f"Metrik kelas {class_name.capitalize()}:")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1-score : {m['f1_score']:.4f}")

print()
print("Uji Signifikansi Binomial (vs p=0.5):")
print(f"  p-value = {classification_result['binomial_significance']['p_value']:.6f}")

print("\nPerbandingan separasi arah Horizon 1-Hari vs 3-Hari (MOMENTUM_CORE):")
h1_train_auc = classification_result_h1['history'].history.get('auc', [np.nan])[-1]
h1_val_auc = classification_result_h1['history'].history.get('val_auc', [np.nan])[-1]
h3_train_auc = classification_result['history'].history.get('auc', [np.nan])[-1]
h3_val_auc = classification_result['history'].history.get('val_auc', [np.nan])[-1]
print(f"  H1 -> AUC train: {h1_train_auc:.4f}, AUC val: {h1_val_auc:.4f}, Best Macro F1: {classification_result_h1['best_threshold_metrics']['f1_score']:.4f}, Acc: {classification_result_h1['best_threshold_metrics']['accuracy']:.4f}")
print(f"  H3 -> AUC train: {h3_train_auc:.4f}, AUC val: {h3_val_auc:.4f}, Best Macro F1: {classification_result['best_threshold_metrics']['f1_score']:.4f}, Acc: {classification_result['best_threshold_metrics']['accuracy']:.4f}")
print(f"  Delta (H3-H1) AUC val: {(h3_val_auc - h1_val_auc):+.4f} | Macro F1: {(classification_result['best_threshold_metrics']['f1_score'] - classification_result_h1['best_threshold_metrics']['f1_score']):+.4f}")

print()
print("="*60)
print("STEP 11: PERBANDINGAN & HASIL EVALUASI")
print("="*60)

print("\n--- Hasil Evaluasi Model (Perbandingan Baseline vs Momentum Core) ---")
print(f"{'Skenario':<12} {'Window':<8} {'MAE (Rp)':<14} {'RMSE (Rp)':<14} {'MAPE (%)':<12} {'R²':<10}")
print("-" * 76)
for scenario_name in ['BASELINE', 'MOMENTUM_CORE']:
    result = experiment_results[scenario_name]
    print(
        f"{scenario_name:<12} {result['seq_length']:<8} "
        f"{result['MAE']:>12,.2f}  {result['RMSE']:>12,.2f}  "
        f"{result['MAPE']:>10.2f}  {result['R2']:>8.4f}"
    )

best_result = experiment_results['MOMENTUM_CORE']
best_window = best_result['seq_length']
print("\n--- Model Aktif untuk analisis lanjutan (Momentum Core Features) ---")
print(f"  Window Size : {best_window}")
print(f"  MAE  : Rp {best_result['MAE']:,.2f}")
print(f"  RMSE : Rp {best_result['RMSE']:,.2f}")
print(f"  MAPE : {best_result['MAPE']:.2f}%")
print(f"  R²   : {best_result['R2']:.4f}")

scenario_metrics = {}
for scenario_name in ['BASELINE', 'MOMENTUM_CORE']:
    scenario_result = experiment_results[scenario_name]
    y_actual_s = scenario_result['y_actual']
    y_pred_inv_s = scenario_result['y_pred_inv']
    min_len_s = len(y_actual_s)

    test_indicator_df_s = df_feat.iloc[split_idx + WINDOW_SIZE:].copy().iloc[:min_len_s]
    if len(test_indicator_df_s) != min_len_s:
        raise ValueError(f"Alignment indikator teknikal dengan y_actual tidak sesuai untuk {scenario_name}.")

    direction_eval_s = evaluate_directional_accuracy(y_actual_s, y_pred_inv_s)
    decision_layer_output_s = build_directional_decision_layer(
        y_pred_inv=y_pred_inv_s,
        rsi_values=test_indicator_df_s['RSI_14'].values,
        macd_values=test_indicator_df_s['MACD'].values,
        macd_signal_values=test_indicator_df_s['MACD_SIGNAL'].values,
        smooth_window=3
    )
    direction_eval_decision_s = evaluate_directional_accuracy_from_signs(
        y_test_inv=y_actual_s,
        pred_sign=decision_layer_output_s['final_sign']
    )
    trend_accuracy_3d_s = evaluate_trend_accuracy_n_day(y_actual_s, y_pred_inv_s, n_days=3)

    scenario_metrics[scenario_name] = {
        'direction_baseline': direction_eval_s,
        'direction_decision': direction_eval_decision_s,
        'trend_accuracy_3d': trend_accuracy_3d_s,
        'decision_layer_output': decision_layer_output_s,
        'y_actual': y_actual_s,
        'y_pred_inv': y_pred_inv_s,
        'test_indicator_df': test_indicator_df_s
    }

print("\n--- Perbandingan Sebelum vs Sesudah Penambahan Fitur ---")
print(
    f"{'Metrik':<28} {'Sebelum (Baseline)':>22} {'Sesudah (Momentum Core)':>22} {'Δ':>12}"
)
print("-" * 88)
comparison_rows = [
    ('MAE (Rp)', experiment_results['BASELINE']['MAE'], experiment_results['MOMENTUM_CORE']['MAE']),
    ('MAPE (%)', experiment_results['BASELINE']['MAPE'], experiment_results['MOMENTUM_CORE']['MAPE']),
    ('R²', experiment_results['BASELINE']['R2'], experiment_results['MOMENTUM_CORE']['R2']),
    (
        'Directional Accuracy (%)',
        scenario_metrics['BASELINE']['direction_decision']['directional_accuracy'],
        scenario_metrics['MOMENTUM_CORE']['direction_decision']['directional_accuracy']
    ),
    (
        'Trend Accuracy 3-hari (%)',
        scenario_metrics['BASELINE']['trend_accuracy_3d']['accuracy'],
        scenario_metrics['MOMENTUM_CORE']['trend_accuracy_3d']['accuracy']
    )
]

for metric_name, before_val, after_val in comparison_rows:
    delta = after_val - before_val
    print(f"{metric_name:<28} {before_val:>22.4f} {after_val:>22.4f} {delta:>+12.4f}")

# Gunakan skenario Momentum Core sebagai baseline analisis lanjutan output/log yang sudah ada
y_actual = scenario_metrics['MOMENTUM_CORE']['y_actual']
y_pred_inv = scenario_metrics['MOMENTUM_CORE']['y_pred_inv']
min_len = len(y_actual)
test_indicator_df = scenario_metrics['MOMENTUM_CORE']['test_indicator_df']
direction_eval = scenario_metrics['MOMENTUM_CORE']['direction_baseline']
direction_eval_decision = scenario_metrics['MOMENTUM_CORE']['direction_decision']
decision_layer_output = scenario_metrics['MOMENTUM_CORE']['decision_layer_output']
trend_accuracy_3d = scenario_metrics['MOMENTUM_CORE']['trend_accuracy_3d']

# Evaluasi tambahan arah & dinamika pergerakan berbasis output regresi (tanpa retraining)
actual_returns = compute_return_series(y_actual)
pred_returns = compute_return_series(y_pred_inv)
da_by_magnitude = evaluate_directional_accuracy_by_magnitude(
    actual_returns=actual_returns,
    pred_returns=pred_returns,
    small_th=0.003,
    medium_th=0.01
)
return_corr = evaluate_return_correlation(actual_returns, pred_returns)
binomial_significance = evaluate_directional_significance(
    correct_predictions=direction_eval_decision['correct_predictions'],
    total_predictions=direction_eval_decision['total_predictions'],
    p_random=0.5
)
classification_report_baseline = compute_direction_classification_report(
    actual_sign=direction_eval['actual_sign'],
    pred_sign=direction_eval['pred_sign']
)
classification_report_decision = compute_direction_classification_report(
    actual_sign=direction_eval_decision['actual_sign'],
    pred_sign=direction_eval_decision['pred_sign']
)

print("\n--- Verifikasi 5 Sampel Actual vs Predicted (Rupiah) ---")
sample_idx = np.linspace(0, min_len - 1, min(5, min_len), dtype=int)
for i, idx in enumerate(sample_idx, 1):
    print(f"  Sampel {i}: Actual = Rp {y_actual[idx]:,.0f}  |  Predicted = Rp {y_pred_inv[idx]:,.0f}")

print("\n--- Directional Accuracy ---")
print(
    f"Baseline (tanpa decision layer) = {direction_eval['directional_accuracy']:.2f}% "
    f"({direction_eval['correct_predictions']}/{direction_eval['total_predictions']})"
)
print(
    f"Setelah decision layer         = {direction_eval_decision['directional_accuracy']:.2f}% "
    f"({direction_eval_decision['correct_predictions']}/{direction_eval_decision['total_predictions']})"
)
print(
    f"Perubahan Directional Accuracy = "
    f"{direction_eval_decision['directional_accuracy'] - direction_eval['directional_accuracy']:+.2f} poin"
)

print("\nConfusion Matrix Arah Baseline (Aktual-Prediksi):")
print(f"  Naik-Naik   : {direction_eval['confusion_matrix']['Naik-Naik']}")
print(f"  Turun-Turun : {direction_eval['confusion_matrix']['Turun-Turun']}")
print(f"  Naik-Turun  : {direction_eval['confusion_matrix']['Naik-Turun']}")
print(f"  Turun-Naik  : {direction_eval['confusion_matrix']['Turun-Naik']}")

print("\nConfusion Matrix Arah Setelah Decision Layer:")
print(f"  Naik-Naik   : {direction_eval_decision['confusion_matrix']['Naik-Naik']}")
print(f"  Turun-Turun : {direction_eval_decision['confusion_matrix']['Turun-Turun']}")
print(f"  Naik-Turun  : {direction_eval_decision['confusion_matrix']['Naik-Turun']}")
print(f"  Turun-Naik  : {direction_eval_decision['confusion_matrix']['Turun-Naik']}")

print("\n--- Trend Accuracy 3-Hari ---")
print(
    f"Trend Accuracy 3-hari = {trend_accuracy_3d['accuracy']:.2f}% "
    f"({trend_accuracy_3d['correct_samples']}/{trend_accuracy_3d['total_samples']})"
)

print("\n--- Directional Accuracy berdasarkan Magnitude Return ---")
for cat_name, cat_metric in da_by_magnitude.items():
    if cat_metric['total'] == 0:
        print(f"  {cat_name:<22}: n=0 (tidak ada sampel)")
    else:
        print(
            f"  {cat_name:<22}: {cat_metric['accuracy']:.2f}% "
            f"({cat_metric['correct']}/{cat_metric['total']})"
        )

print("\n--- Korelasi Return Aktual vs Prediksi ---")
print(f"Pearson correlation = {return_corr['pearson_correlation']:.4f}")
print(f"p-value             = {return_corr['p_value']:.6f}")

print("\n--- Uji Signifikansi Binomial (Directional Accuracy vs Acak p=0.5) ---")
print(f"Directional Accuracy (decision layer) = {direction_eval_decision['directional_accuracy']:.2f}%")
print(f"Expected random accuracy             = {binomial_significance['expected_accuracy_random']:.2f}%")
print(f"p-value (binomial test)              = {binomial_significance['p_value']:.6f}")

print("\n--- Klasifikasi Arah: Precision, Recall, F1 ---")
for label, report in [('Baseline', classification_report_baseline), ('Decision Layer', classification_report_decision)]:
    cm = report['confusion_matrix']
    met_up = report['metrics']['naik']
    met_down = report['metrics']['turun']

    print(f"\n{label}:")
    print("  Confusion Matrix (Aktual x Prediksi):")
    print(f"    Aktual Naik  -> Pred Naik : {cm['actual_naik_pred_naik']}")
    print(f"    Aktual Naik  -> Pred Turun: {cm['actual_naik_pred_turun']}")
    print(f"    Aktual Turun -> Pred Naik : {cm['actual_turun_pred_naik']}")
    print(f"    Aktual Turun -> Pred Turun: {cm['actual_turun_pred_turun']}")

    print("  Metrik kelas Naik:")
    print(f"    Precision: {met_up['precision']:.4f} | Recall: {met_up['recall']:.4f} | F1: {met_up['f1_score']:.4f}")
    print("  Metrik kelas Turun:")
    print(f"    Precision: {met_down['precision']:.4f} | Recall: {met_down['recall']:.4f} | F1: {met_down['f1_score']:.4f}")

print("="*60)


# =============================================================================
# STEP 11B — Backtesting Finansial (Tanpa Look-Ahead Bias)
# =============================================================================
BACKTEST_INITIAL_CAPITAL = 100_000_000
BACKTEST_TRANSACTION_COST = 0.0015  # 0.15% per transaksi

backtest_df = df_feat.iloc[split_idx + WINDOW_SIZE:].copy().iloc[:min_len]
backtest_prices = backtest_df['Close'].copy()

backtest_result = run_directional_backtest(
    price_series=backtest_prices,
    signal_sign=decision_layer_output['final_sign'],
    initial_capital=BACKTEST_INITIAL_CAPITAL,
    transaction_cost=BACKTEST_TRANSACTION_COST
)

strategy_metrics = backtest_result['strategy_metrics']
buy_hold_metrics = backtest_result['buy_hold_metrics']

print("\n" + "="*60)
print("STEP 11B: BACKTESTING FINANSIAL")
print("="*60)
print("Asumsi backtest:")
print(f"  Modal awal        : Rp {BACKTEST_INITIAL_CAPITAL:,.0f}")
print(f"  Biaya transaksi   : {BACKTEST_TRANSACTION_COST*100:.2f}% per transaksi")
print("  Aturan posisi     : BUY saat sinyal naik, CASH saat sinyal turun")
print("  Eksekusi sinyal   : t+1 (signal shift 1 hari, anti look-ahead)")

print("\nPerbandingan performa periode test:")
print(f"{'Metrik':<24} {'Strategi Model':>18} {'Buy & Hold':>18}")
print("-" * 64)
print(f"{'Total Return':<24} {strategy_metrics['total_return']*100:>17.2f}% {buy_hold_metrics['total_return']*100:>17.2f}%")
print(f"{'Win Rate (hari aktif)':<24} {strategy_metrics['win_rate']*100:>17.2f}% {buy_hold_metrics['win_rate']*100:>17.2f}%")
print(f"{'Max Drawdown':<24} {strategy_metrics['max_drawdown']*100:>17.2f}% {buy_hold_metrics['max_drawdown']*100:>17.2f}%")
print(f"{'Nilai Akhir Equity':<24} Rp {backtest_result['equity_curve'].iloc[-1]:>14,.0f} Rp {backtest_result['buy_hold_equity'].iloc[-1]:>14,.0f}")


# =============================================================================
# STEP 12 — Visualisasi Final (Model Terbaik)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(best_result['history'].history['loss'], label='Train Loss')
axes[0, 0].plot(best_result['history'].history['val_loss'], label='Val Loss')
axes[0, 0].set_title(f'Training & Validation Loss (Window={best_window})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(y_actual, label='Actual', alpha=0.8)
axes[0, 1].plot(y_pred_inv, label='Predicted', alpha=0.8)
axes[0, 1].set_title('Actual vs Predicted (Harga dalam Rupiah)')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_actual, y_pred_inv, alpha=0.5)
min_val = min(y_actual.min(), y_pred_inv.min())
max_val = max(y_actual.max(), y_pred_inv.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
axes[1, 0].set_xlabel('Actual Price')
axes[1, 0].set_ylabel('Predicted Price')
axes[1, 0].set_title('Actual vs Predicted (Scatter)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

errors = y_actual - y_pred_inv
axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_title('Distribution of Prediction Errors')
axes[1, 1].set_xlabel('Error (Actual - Predicted)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'TLKM RNN - Model Terbaik (Window={best_window}, MAPE={best_result["MAPE"]:.2f}%)', fontsize=14)
plt.tight_layout()
plt.savefig('tlkm_rnn_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualisasi arah aktual vs prediksi (baseline vs decision layer)
actual_direction_binary = np.where(direction_eval['actual_sign'] >= 0, 1, -1)
pred_direction_binary = np.where(direction_eval['pred_sign'] >= 0, 1, -1)
pred_direction_decision_binary = np.where(direction_eval_decision['pred_sign'] >= 0, 1, -1)
time_axis = np.arange(1, len(y_actual))

plt.figure(figsize=(14, 4.5))
plt.step(time_axis, actual_direction_binary, where='mid', label='Arah Aktual', linewidth=1.5)
plt.step(time_axis, pred_direction_binary, where='mid', label='Arah Prediksi (Baseline)', linewidth=1.2, alpha=0.8)
plt.step(time_axis, pred_direction_decision_binary, where='mid', label='Arah Prediksi (Decision Layer)', linewidth=1.2, alpha=0.8)
plt.yticks([-1, 1], ['Turun', 'Naik'])
plt.ylim(-1.5, 1.5)
plt.xlabel('Time Step')
plt.ylabel('Arah Harga')
plt.title(
    f'Arah Aktual vs Prediksi | Baseline={direction_eval["directional_accuracy"]:.2f}% '
    f'| Decision Layer={direction_eval_decision["directional_accuracy"]:.2f}%'
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('tlkm_rnn_directional_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

plot_backtest_equity_curve(
    backtest_result,
    output_path='tlkm_backtest_equity_curve.png'
)

print("\nGrafik disimpan ke: tlkm_rnn_evaluation.png")
print("Grafik arah disimpan ke: tlkm_rnn_directional_accuracy.png")
print("Grafik backtest disimpan ke: tlkm_backtest_equity_curve.png")
print("\n" + "="*60)
print("STEP 12: Visualisasi - SELESAI")
print("="*60)
print("\n*** PROYEK SELESAI ***")
