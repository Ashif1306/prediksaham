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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import os

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
    
    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available = [c for c in required_cols if c in df.columns]
    if 'Close' not in available:
        # Multi-level columns dari yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        available = [c for c in required_cols if c in df.columns]
    
    df = df[available]
    
    # Bersihkan index agar valid datetime
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]

    # Pastikan kolom numerik (fallback file lokal kadang terbaca sebagai string)
    for col in available:
        df[col] = pd.to_numeric(df[col], errors='coerce')

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
# STEP 5 — Feature Engineering Stabil
# =============================================================================
def create_features(df):
    """Feature engineering yang stabil."""
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
    
    # Hapus baris dengan NaN dari rolling
    df = df.dropna()
    
    return df

df_feat = create_features(df_clean)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'MACD_SIGNAL']
feature_cols = [c for c in feature_cols if c in df_feat.columns]

print(f"\nFitur yang digunakan: {feature_cols}")
print(f"Baris setelah feature engineering: {len(df_feat)}")

print("\n" + "="*60)
print("STEP 5: Feature Engineering - SELESAI")
print("="*60)


# =============================================================================
# STEP 6 — Normalisasi TANPA BUG (PENTING)
# =============================================================================
# PENTING: Scaler HANYA di-fit pada data TRAINING untuk menghindari data leakage
# Scaler akan di-fit nanti setelah train-test split

TARGET_COL = 'Close'
data_for_scale = df_feat[feature_cols].values

# Buat scaler global - akan di-fit pada TRAIN saja di Step 8
scaler_global = MinMaxScaler(feature_range=(0, 1))

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
        y.append(data[i + seq_length, target_idx])  # Target = kolom Close
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
train_data = df_feat.iloc[:split_idx][feature_cols].values
test_data = df_feat.iloc[split_idx:][feature_cols].values

scaler_global.fit(train_data)
train_scaled = scaler_global.transform(train_data)
test_scaled = scaler_global.transform(test_data)

print(f"\nTrain data: {train_data.shape[0]} baris | Test data: {test_data.shape[0]} baris")

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

def run_experiment(seq_length, train_scaled, test_scaled, scaler_global, feature_cols, close_idx, seed=SEED):
    """Jalankan eksperimen untuk satu window size."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    X_train, y_train = create_sequences(train_scaled, seq_length, close_idx)
    X_test, y_test = create_sequences(test_scaled, seq_length, close_idx)
    
    n_features = X_train.shape[2]
    model = build_model(seq_length, n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
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
    y_actual = inverse_transform_close(scaler_global, y_test, close_idx)
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

# Jalankan eksperimen untuk window 10
WINDOW_SIZE = 10

print("\n" + "="*60)
print("STEP 9-10: EKSPERIMEN (Window 10)")
print("="*60)

print(f"\n>>> Melatih model dengan window size = {WINDOW_SIZE} ...")
close_idx = feature_cols.index(TARGET_COL)
best_result = run_experiment(WINDOW_SIZE, train_scaled, test_scaled, scaler_global, feature_cols, close_idx)
best_window = best_result['seq_length']
print(f"    MAPE = {best_result['MAPE']:.2f}% | MAE = Rp {best_result['MAE']:,.2f} | R² = {best_result['R2']:.4f}")

print("\n" + "="*60)
print("STEP 11: PERBANDINGAN & HASIL EVALUASI")
print("="*60)

print("\n--- Hasil Evaluasi Model (Fokus Stabilitas Generalisasi) ---")
print(f"{'Window':<10} {'MAE (Rp)':<14} {'RMSE (Rp)':<14} {'MAPE (%)':<12} {'R²':<10}")
print("-" * 60)
print(f"{best_result['seq_length']:<10} {best_result['MAE']:>12,.2f}  {best_result['RMSE']:>12,.2f}  {best_result['MAPE']:>10.2f}  {best_result['R2']:>8.4f}")

print("\n--- Model Terbaik (MAPE Terendah) ---")
print(f"  Window Size : {best_window}")
print(f"  MAE  : Rp {best_result['MAE']:,.2f}")
print(f"  RMSE : Rp {best_result['RMSE']:,.2f}")
print(f"  MAPE : {best_result['MAPE']:.2f}%")
print(f"  R²   : {best_result['R2']:.4f}")

# Verifikasi 5 sampel (model terbaik)
y_actual = best_result['y_actual']
y_pred_inv = best_result['y_pred_inv']
min_len = len(y_actual)

direction_eval = evaluate_directional_accuracy(y_actual, y_pred_inv)

print("\n--- Verifikasi 5 Sampel Actual vs Predicted (Rupiah) ---")
sample_idx = np.linspace(0, min_len - 1, min(5, min_len), dtype=int)
for i, idx in enumerate(sample_idx, 1):
    print(f"  Sampel {i}: Actual = Rp {y_actual[idx]:,.0f}  |  Predicted = Rp {y_pred_inv[idx]:,.0f}")

print("\n--- Directional Accuracy ---")
print(
    f"Directional Accuracy = {direction_eval['directional_accuracy']:.2f}% "
    f"({direction_eval['correct_predictions']}/{direction_eval['total_predictions']})"
)

print("\nConfusion Matrix Arah (Aktual-Prediksi):")
print(f"  Naik-Naik   : {direction_eval['confusion_matrix']['Naik-Naik']}")
print(f"  Turun-Turun : {direction_eval['confusion_matrix']['Turun-Turun']}")
print(f"  Naik-Turun  : {direction_eval['confusion_matrix']['Naik-Turun']}")
print(f"  Turun-Naik  : {direction_eval['confusion_matrix']['Turun-Naik']}")
print("="*60)


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

# Visualisasi arah aktual vs prediksi (mengikuti definisi Directional Accuracy)
actual_direction_binary = np.where(direction_eval['actual_sign'] >= 0, 1, -1)
pred_direction_binary = np.where(direction_eval['pred_sign'] >= 0, 1, -1)
time_axis = np.arange(1, len(y_actual))

plt.figure(figsize=(14, 4.5))
plt.step(time_axis, actual_direction_binary, where='mid', label='Arah Aktual', linewidth=1.5)
plt.step(time_axis, pred_direction_binary, where='mid', label='Arah Prediksi', linewidth=1.2, alpha=0.8)
plt.yticks([-1, 1], ['Turun', 'Naik'])
plt.ylim(-1.5, 1.5)
plt.xlabel('Time Step')
plt.ylabel('Arah Harga')
plt.title(
    f'Arah Aktual vs Prediksi | Directional Accuracy = {direction_eval["directional_accuracy"]:.2f}%'
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('tlkm_rnn_directional_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nGrafik disimpan ke: tlkm_rnn_evaluation.png")
print("Grafik arah disimpan ke: tlkm_rnn_directional_accuracy.png")
print("\n" + "="*60)
print("STEP 12: Visualisasi - SELESAI")
print("="*60)
print("\n*** PROYEK SELESAI ***")
