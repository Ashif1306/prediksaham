"""
Proyek Prediksi Harga Saham PT Telkom Indonesia (TLKM.JK)
Menggunakan Recurrent Neural Network (RNN) - Versi Clean
Data: Yahoo Finance 2008 - Dinamis
Focus: Regression Model
"""

import sys
if sys.version_info >= (3, 14):
    print("="*70)
    print("ERROR: TensorFlow belum mendukung Python 3.14")
    print("Gunakan Python 3.11 atau 3.12.")
    print("="*70)
    sys.exit(1)

# =============================================================================
# STEP 1 — Setup Environment
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seed untuk reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*70)
print("STEP 1: Setup Environment")
print("="*70)
print(f"TensorFlow version : {tf.__version__}")
print(f"Pandas version     : {pd.__version__}")
print(f"NumPy version      : {np.__version__}")
print(f"Random seed        : {SEED}")
print("="*70)


# =============================================================================
# STEP 2 — Ambil Data TLKM
# =============================================================================
def load_data():
    """Mengunduh data historis TLKM dari Yahoo Finance."""
    ticker = "TLKM.JK"
    start_date = "2008-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\nMengunduh data TLKM dari Yahoo Finance...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, 
                        progress=False, auto_adjust=False)
        
        if df.empty or len(df) < 100:
            raise ValueError(f"Data tidak cukup: hanya {len(df)} baris")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        print(f"✓ Data berhasil diunduh")
        print(f"  Periode : {df.index[0].date()} hingga {df.index[-1].date()}")
        print(f"  Total   : {len(df)} hari trading")
        
        # Simpan ke CSV
        output_file = "data_tlkm_harga_saham.csv"
        df.to_csv(output_file)
        print(f"  File    : {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        # Fallback ke file lokal
        if os.path.exists("data_tlkm_harga_saham.csv"):
            df = pd.read_csv("data_tlkm_harga_saham.csv", 
                           index_col=0, parse_dates=True)
            print("  Menggunakan data lokal: data_tlkm_harga_saham.csv")
        else:
            raise
    
    return df


print("\n" + "="*70)
print("STEP 2: Pengambilan Data")
print("="*70)
df_raw = load_data()
print("="*70)


# =============================================================================
# STEP 3 — Data Cleaning
# =============================================================================
def clean_data(df):
    """Pembersihan data OHLCV dengan penanganan MultiIndex."""
    df = df.copy()
    
    # Flatten MultiIndex columns jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(part).strip() for part in col_tuple 
                              if part and str(part).strip())
                     for col_tuple in df.columns.to_flat_index()]
    
    # Deteksi kolom OHLCV
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    selected_cols = {}
    
    for base_col in required_cols:
        candidates = [col for col in df.columns 
                     if col == base_col or col.startswith(f'{base_col}_')]
        if candidates:
            selected_cols[base_col] = candidates[0]
    
    if not selected_cols:
        raise ValueError(f"Kolom OHLCV tidak ditemukan. "
                        f"Tersedia: {list(df.columns)}")
    
    # Pilih dan rename kolom
    df = df[list(selected_cols.values())].copy()
    df = df.rename(columns={v: k for k, v in selected_cols.items()})
    
    # Hapus duplikat kolom
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T
    
    # Validasi dan konversi datetime index
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='first')]
    
    # Konversi ke numerik
    df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    
    # Isi missing values
    df = df.ffill().bfill()
    df = df.dropna()
    
    # Validasi: hapus harga negatif atau nol
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df = df[df[col] > 0]
    
    return df


print("\n" + "="*70)
print("STEP 3: Data Cleaning")
print("="*70)
df_clean = clean_data(df_raw)
print(f"✓ Data cleaning selesai")
print(f"  Baris sebelum : {len(df_raw)}")
print(f"  Baris sesudah : {len(df_clean)}")
print(f"  Missing values: {df_clean.isna().sum().sum()}")
print("="*70)


# =============================================================================
# STEP 4 — EDA Dasar
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Exploratory Data Analysis")
print("="*70)

print("\nStatistik Deskriptif:")
print(df_clean.describe().round(2))

print("\nKorelasi antar fitur:")
corr_matrix = df_clean[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
print(corr_matrix.round(3))

print("\nInfo Dataset:")
print(f"  Shape     : {df_clean.shape}")
print(f"  Columns   : {list(df_clean.columns)}")
print(f"  Date Range: {df_clean.index[0].date()} to {df_clean.index[-1].date()}")
print("="*70)


# =============================================================================
# STEP 5 — Feature Engineering
# =============================================================================
def create_features(df):
    """Membuat fitur teknikal untuk model regresi."""
    df = df.copy()
    
    if 'Close' not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan")
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Momentum features
    df['RETURN_1D'] = df['Close'].pct_change()
    df['RETURN_LAG_1'] = df['RETURN_1D'].shift(1)
    df['RETURN_LAG_2'] = df['RETURN_1D'].shift(2)
    df['RSI_SLOPE'] = df['RSI_14'].diff()
    df['ROLL_STD_RETURN_5D'] = df['RETURN_1D'].rolling(
        window=5, min_periods=5).std()
    
    # Hapus baris dengan NaN
    df = df.dropna()
    
    return df


print("\n" + "="*70)
print("STEP 5: Feature Engineering")
print("="*70)

df_feat = create_features(df_clean)

# Fitur yang akan digunakan
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RETURN_LAG_1', 'RETURN_LAG_2', 
    'RSI_SLOPE', 'ROLL_STD_RETURN_5D'
]
TARGET_COL = 'Close'

print(f"✓ Feature engineering selesai")
print(f"  Baris sebelum : {len(df_clean)}")
print(f"  Baris sesudah : {len(df_feat)}")
print(f"  Fitur yang digunakan ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:2d}. {col}")
print(f"\n  Target kolom  : {TARGET_COL}")
print("="*70)


# =============================================================================
# STEP 6 — Normalisasi (Fit HANYA pada Train Set)
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Normalisasi dengan MinMaxScaler")
print("="*70)
print("⚠ PENTING: Scaler akan di-fit HANYA pada data training")
print("           untuk menghindari data leakage")
print("           (Fitting dilakukan di STEP 8)")
print("="*70)


# =============================================================================
# STEP 7 — Sliding Window
# =============================================================================
def create_sequences(data, seq_length, target_idx):
    """
    Membuat sequence time series dengan sliding window.
    
    Args:
        data: Array yang sudah dinormalisasi (samples, features)
        seq_length: Panjang sequence (timesteps)
        target_idx: Index kolom target dalam array
    
    Returns:
        X: (samples, timesteps, features)
        y: (samples,) - nilai target pada timestep berikutnya
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)


SEQ_LENGTH = 10

print("\n" + "="*70)
print("STEP 7: Sliding Window Configuration")
print("="*70)
print(f"✓ Sequence length (timesteps): {SEQ_LENGTH}")
print(f"  Input : {SEQ_LENGTH} hari historis")
print(f"  Output: Prediksi harga hari ke-{SEQ_LENGTH + 1}")
print("="*70)


# =============================================================================
# STEP 8 — Train-Test Split (Tanpa Shuffle, Time Series)
# =============================================================================
def scale_data(df_features, feature_cols, train_ratio=0.8):
    """
    Split dan scale data dengan fit HANYA pada train set.
    
    Returns:
        dict dengan train_scaled, test_scaled, scaler, split_idx, close_idx
    """
    split_idx = int(len(df_features) * train_ratio)
    
    # Split data
    train_data = df_features.iloc[:split_idx][feature_cols].values
    test_data = df_features.iloc[split_idx:][feature_cols].values
    
    # Fit scaler HANYA pada train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    # Transform kedua set
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Cari index kolom Close
    close_idx = feature_cols.index('Close')
    
    return {
        'train_scaled': train_scaled,
        'test_scaled': test_scaled,
        'scaler': scaler,
        'split_idx': split_idx,
        'close_idx': close_idx
    }


TRAIN_RATIO = 0.8

print("\n" + "="*70)
print("STEP 8: Train-Test Split & Normalization")
print("="*70)

data_dict = scale_data(df_feat, FEATURE_COLS, TRAIN_RATIO)

print(f"✓ Data split selesai (tanpa shuffle)")
print(f"  Train ratio   : {TRAIN_RATIO*100:.0f}%")
print(f"  Split index   : {data_dict['split_idx']}")
print(f"  Train samples : {data_dict['train_scaled'].shape[0]}")
print(f"  Test samples  : {data_dict['test_scaled'].shape[0]}")
print(f"\n✓ Normalisasi selesai (MinMaxScaler [0,1])")
print(f"  Scaler fitted : Train set only (NO LEAKAGE)")
print(f"  Target index  : {data_dict['close_idx']} (kolom {TARGET_COL})")
print("="*70)


# =============================================================================
# STEP 9 — Build Model
# =============================================================================
def build_model(seq_length, n_features):
    """Arsitektur SimpleRNN untuk regresi harga saham."""
    model = Sequential([
        SimpleRNN(64, return_sequences=True, 
                 input_shape=(seq_length, n_features)),
        Dropout(0.25),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dense(1)  # Output: prediksi harga
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model


print("\n" + "="*70)
print("STEP 9: Build Model Architecture")
print("="*70)

# Buat sequences
X_train, y_train = create_sequences(
    data_dict['train_scaled'], 
    SEQ_LENGTH, 
    data_dict['close_idx']
)
X_test, y_test = create_sequences(
    data_dict['test_scaled'], 
    SEQ_LENGTH, 
    data_dict['close_idx']
)

n_features = X_train.shape[2]
model = build_model(SEQ_LENGTH, n_features)

print("✓ Model SimpleRNN berhasil dibuat")
print(f"\n  Input shape    : {X_train.shape}")
print(f"  - Samples      : {X_train.shape[0]}")
print(f"  - Timesteps    : {X_train.shape[1]}")
print(f"  - Features     : {X_train.shape[2]}")
print(f"\n  Output shape   : {y_train.shape}")
print(f"  - Target       : Harga saham (continuous)")

print("\n  Arsitektur Model:")
print("  ┌" + "─"*50 + "┐")
model.summary(print_fn=lambda x: print("  │ " + x.ljust(48) + "│"))
print("  └" + "─"*50 + "┘")
print("="*70)


# =============================================================================
# STEP 10 — Training Model
# =============================================================================
def train_model(model, X_train, y_train, X_test, y_test, epochs=150):
    """Training model dengan callbacks."""
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        verbose=1
    )
    
    print("\n🚀 Memulai training...")
    print("  Callbacks aktif:")
    print("    • EarlyStopping (patience=20)")
    print("    • ReduceLROnPlateau (patience=6, factor=0.5)")
    print()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        shuffle=False,  # PENTING: jangan shuffle time series
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history


print("\n" + "="*70)
print("STEP 10: Training Model")
print("="*70)

history = train_model(model, X_train, y_train, X_test, y_test)

print("\n✓ Training selesai!")
print(f"  Epochs trained : {len(history.history['loss'])}")
print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss  : {history.history['val_loss'][-1]:.6f}")
print("="*70)


# =============================================================================
# STEP 11 — Evaluasi Regresi
# =============================================================================
def inverse_transform_close(scaler, values, close_idx):
    """Inverse transform untuk kolom Close."""
    values = np.asarray(values)
    return (values - scaler.min_[close_idx]) / scaler.scale_[close_idx]


def evaluate_regression(model, X_test, y_test, scaler, close_idx):
    """Evaluasi performa model regresi."""
    # Prediksi
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_actual_scaled = y_test
    
    # Inverse transform ke harga asli
    y_actual = inverse_transform_close(scaler, y_actual_scaled, close_idx)
    y_pred = inverse_transform_close(scaler, y_pred_scaled, close_idx)
    
    # Pastikan panjang sama
    min_len = min(len(y_actual), len(y_pred))
    y_actual = y_actual[:min_len]
    y_pred = y_pred[:min_len]
    
    # Hitung metrik
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-8))) * 100
    
    return {
        'y_actual': y_actual,
        'y_pred': y_pred,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def calculate_trend_accuracy_3day(y_actual, y_pred, n_days=3):
    """
    Hitung akurasi arah tren berdasarkan return kumulatif n-hari.
    Derived direction dari hasil regresi, bukan classifier terpisah.
    """
    y_actual = np.asarray(y_actual).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_actual) <= n_days:
        raise ValueError(f"Minimal {n_days+1} data untuk trend {n_days}-hari")
    
    # Return kumulatif n-hari
    actual_cumret = (y_actual[n_days:] / (y_actual[:-n_days] + 1e-12)) - 1.0
    pred_cumret = (y_pred[n_days:] / (y_pred[:-n_days] + 1e-12)) - 1.0
    
    # Arah tren
    actual_sign = np.sign(actual_cumret)
    pred_sign = np.sign(pred_cumret)
    
    # Akurasi
    correct = np.sum(actual_sign == pred_sign)
    total = len(actual_sign)
    accuracy = (correct / total) * 100
    
    return {
        'n_days': n_days,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


print("\n" + "="*70)
print("STEP 11: Evaluasi Model Regresi")
print("="*70)

eval_results = evaluate_regression(
    model, X_test, y_test, 
    data_dict['scaler'], 
    data_dict['close_idx']
)

print("\n📊 Metrik Regresi:")
print(f"  MAE (Mean Absolute Error)       : Rp {eval_results['MAE']:,.2f}")
print(f"  RMSE (Root Mean Squared Error)  : Rp {eval_results['RMSE']:,.2f}")
print(f"  MAPE (Mean Absolute % Error)    : {eval_results['MAPE']:.2f}%")
print(f"  R² Score (Coefficient of Det.)  : {eval_results['R2']:.4f}")

print("\n📈 Analisis Tambahan - Trend Accuracy:")
trend_results = calculate_trend_accuracy_3day(
    eval_results['y_actual'], 
    eval_results['y_pred'], 
    n_days=3
)
print(f"  Trend Accuracy 3-hari : {trend_results['accuracy']:.2f}%")
print(f"  (Prediksi arah benar  : {trend_results['correct']}/{trend_results['total']})")
print(f"  * Derived dari hasil regresi (bukan classifier terpisah)")

print("\n📝 Sample Predictions (5 titik):")
sample_indices = np.linspace(0, len(eval_results['y_actual'])-1, 5, dtype=int)
print(f"  {'Index':<8} {'Actual (Rp)':<15} {'Predicted (Rp)':<15} {'Error (Rp)':<15}")
print("  " + "-"*55)
for idx in sample_indices:
    actual = eval_results['y_actual'][idx]
    pred = eval_results['y_pred'][idx]
    error = actual - pred
    print(f"  {idx:<8} {actual:>13,.0f}  {pred:>13,.0f}  {error:>13,.0f}")

print("="*70)


# =============================================================================
# STEP 12 — Visualisasi
# =============================================================================
def plot_results(history, y_actual, y_pred, mape, r2, output_file='tlkm_rnn_results.png'):
    """Visualisasi hasil training dan prediksi."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training & Validation Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted (Time Series)
    axes[0, 1].plot(y_actual, label='Actual', alpha=0.8, linewidth=1.5)
    axes[0, 1].plot(y_pred, label='Predicted', alpha=0.8, linewidth=1.5)
    axes[0, 1].set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Harga (Rp)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter Plot
    axes[1, 0].scatter(y_actual, y_pred, alpha=0.5, s=20)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Price (Rp)')
    axes[1, 0].set_ylabel('Predicted Price (Rp)')
    axes[1, 0].set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error Distribution
    errors = y_actual - y_pred
    axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Error (Actual - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle(f'TLKM Stock Price Prediction - SimpleRNN\n'
                f'MAPE: {mape:.2f}% | R²: {r2:.4f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("\nSTEP 1 — Setup Environment")
    print(f"Seed            : {SEED}")
    print(f"Window Size     : {WINDOW_SIZE}")
    print(f"Train Ratio     : {TRAIN_RATIO}")
    print(f"Learning Rate   : {LEARNING_RATE}")

    print("\nSTEP 2 — Ambil Data")
    df_raw = load_data()
    print(f"Data mentah berhasil dimuat: {df_raw.shape}")

    print("\nSTEP 3 — Data Cleaning")
    df_clean = clean_data(df_raw)
    print(f"Data setelah cleaning: {df_clean.shape}")

    print("\nSTEP 4 — EDA Dasar")
    run_basic_eda(df_clean)

    print("\nSTEP 5 — Feature Engineering")
    df_features = create_features(df_clean)
    print(f"Data fitur final: {df_features.shape}")
    print(f"Total fitur     : {len(df_features.columns)}")

    print("\nSTEP 6 — Normalisasi (fit hanya pada train set, tanpa leakage)")
    print("Menerapkan MinMaxScaler: fit pada train, transform train+test.")

    print("\nSTEP 8 — Train-Test Split tanpa shuffle")
    print("Split data dilakukan secara kronologis (tanpa shuffle).")

    train_scaled, test_scaled, scaler, split_idx = scale_data(df_features, train_ratio=TRAIN_RATIO)
    print(f"Split index      : {split_idx}")
    print(f"Train shape      : {train_scaled.shape}")
    print(f"Test shape       : {test_scaled.shape}")

    target_idx = df_features.columns.get_loc(TARGET_COL)

    print("\nSTEP 7 — Sliding Window")
    X_train, y_train = create_sequences(train_scaled, window_size=WINDOW_SIZE, target_idx=target_idx)
    X_test, y_test = create_sequences(test_scaled, window_size=WINDOW_SIZE, target_idx=target_idx)
    print(f"X_train shape    : {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test shape     : {X_test.shape}, y_test : {y_test.shape}")

    print("\nSTEP 9 — Build Model")
    model = build_model(window_size=WINDOW_SIZE, n_features=X_train.shape[2], learning_rate=LEARNING_RATE)
    print("Model SimpleRNN berhasil dibangun.")

    print("\nSTEP 10 — Training")
    train_model(model, X_train, y_train)
    print("Training selesai.")

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)

    y_test_actual = inverse_close(y_test, scaler, close_idx=target_idx)
    y_pred_actual = inverse_close(y_pred_scaled, scaler, close_idx=target_idx)

    print("\nSTEP 11 — Evaluasi Regresi")
    metrics = evaluate_regression(y_test_actual, y_pred_actual)
    trend_acc_3d = calculate_trend_accuracy_3day(y_test_actual, y_pred_actual)

    print("\n=== HASIL EVALUASI REGRESI ===")
    print(f"MAE   : {metrics['MAE']:.4f}")
    print(f"RMSE  : {metrics['RMSE']:.4f}")
    print(f"MAPE  : {metrics['MAPE']:.2f}%")
    print(f"R2    : {metrics['R2']:.4f}")
    print(f"Trend Accuracy 3-hari: {trend_acc_3d:.2f}%")

    print("\nSTEP 12 — Visualisasi")
    test_index = df_features.index[split_idx + WINDOW_SIZE :]
    plot_results(test_index, y_test_actual, y_pred_actual, output_path="tlkm_rnn_evaluation.png")
    print("Grafik evaluasi disimpan ke: tlkm_rnn_evaluation.png")


if __name__ == "__main__":
    main()
