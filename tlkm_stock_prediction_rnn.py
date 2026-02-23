"""
Proyek Prediksi Harga Saham PT Telkom Indonesia (TLKM.JK)
Menggunakan Recurrent Neural Network (RNN)
Data: Yahoo Finance 2008 - Dinamis
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
import joblib
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

# Jalankan script dari root folder project agar path relatif berikut valid.
DATA_CSV_PATH = "data/data_tlkm_harga_saham.csv"
DATA_FEAT_CSV_PATH = "data/data_tlkm_features.csv"
MODEL_PATH = "models/tlkm_rnn_model.keras"
SCALER_PATH = "models/tlkm_scaler.pkl"
PLOT_PATH = "img/tlkm_rnn_results.png"

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
        
        os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
        df.to_csv(DATA_CSV_PATH)
        print(f"  File    : {DATA_CSV_PATH}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if os.path.exists(DATA_CSV_PATH):
            df = pd.read_csv(DATA_CSV_PATH, index_col=0, parse_dates=True)
            print(f"  Menggunakan data lokal: {DATA_CSV_PATH}")
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
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(part).strip() for part in col_tuple 
                              if part and str(part).strip())
                     for col_tuple in df.columns.to_flat_index()]
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    selected_cols = {}
    
    for base_col in required_cols:
        candidates = [col for col in df.columns 
                     if col == base_col or col.startswith(f'{base_col}_')]
        if candidates:
            selected_cols[base_col] = candidates[0]
    
    if not selected_cols:
        raise ValueError(f"Kolom OHLCV tidak ditemukan. Tersedia: {list(df.columns)}")
    
    df = df[list(selected_cols.values())].copy()
    df = df.rename(columns={v: k for k, v in selected_cols.items()})
    
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T
    
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='first')]
    df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    df = df.ffill().bfill()
    df = df.dropna()
    
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
    """
    Membuat 11 fitur teknikal untuk model RNN.

    Fitur yang dihasilkan (selain OHLCV):
      - RETURN_LAG_1    : Return 1 hari sebelumnya (momentum sangat pendek)
      - RETURN_LAG_2    : Return 2 hari sebelumnya (konteks tambahan)
      - RSI_SLOPE       : Kemiringan RSI-14 (perubahan kekuatan momentum)
      - ROLL_STD_RETURN_5D : Volatilitas return 5 hari (risiko jangka pendek)
      - MA_5            : Moving Average 5 hari (tren jangka pendek)
      - MA_10           : Moving Average 10 hari (tren jangka menengah)

    Kolom intermediate (RSI_14, MACD, dll.) dihitung sementara lalu dibuang
    sehingga df_feat hanya berisi 11 kolom final.
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        raise ValueError("Kolom Close tidak ditemukan")
    
    # ── Intermediate: RSI-14 (dibutuhkan untuk RSI_SLOPE) ─────────────────
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    rsi_14   = 100 - (100 / (1 + rs))

    # ── Intermediate: Return harian (dibutuhkan untuk LAG & ROLL_STD) ──────
    return_1d = df['Close'].pct_change()

    # ── 6 Fitur Teknikal Final ─────────────────────────────────────────────
    df['RETURN_LAG_1']      = return_1d.shift(1)
    df['RETURN_LAG_2']      = return_1d.shift(2)
    df['RSI_SLOPE']         = rsi_14.diff()
    df['ROLL_STD_RETURN_5D'] = return_1d.rolling(window=5, min_periods=5).std()
    df['MA_5']              = df['Close'].rolling(window=5,  min_periods=5).mean()
    df['MA_10']             = df['Close'].rolling(window=10, min_periods=10).mean()

    # Hapus baris dengan NaN (akibat rolling & shift)
    df = df.dropna()

    # Pastikan hanya 11 kolom final yang tersimpan
    FINAL_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RETURN_LAG_1', 'RETURN_LAG_2',
        'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
        'MA_5', 'MA_10'
    ]
    df = df[FINAL_COLS]

    return df


print("\n" + "="*70)
print("STEP 5: Feature Engineering")
print("="*70)

df_feat = create_features(df_clean)

# Fitur yang akan digunakan (11 fitur) — identik dengan kolom df_feat
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RETURN_LAG_1', 'RETURN_LAG_2', 
    'RSI_SLOPE', 'ROLL_STD_RETURN_5D',
    'MA_5', 'MA_10'
]
TARGET_COL = 'Close'

print(f"✓ Feature engineering selesai")
print(f"  Baris sebelum : {len(df_clean)}")
print(f"  Baris sesudah : {len(df_feat)}")
print(f"  Kolom df_feat : {df_feat.shape[1]}  ← tepat 11, tidak ada kolom intermediate")
print(f"\n  Fitur yang digunakan ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:2d}. {col}")
print(f"\n  Target kolom  : {TARGET_COL}")

# ── Simpan df_feat ke CSV ──────────────────────────────────────────────────
os.makedirs(os.path.dirname(DATA_FEAT_CSV_PATH), exist_ok=True)
df_feat.to_csv(DATA_FEAT_CSV_PATH)
print(f"\n✓ Feature DataFrame disimpan ke  : {DATA_FEAT_CSV_PATH}")
print(f"  Shape  : {df_feat.shape}")
print(f"  Kolom  : {list(df_feat.columns)}")
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
    split_idx = int(len(df_features) * train_ratio)
    
    train_data = df_features.iloc[:split_idx][feature_cols].values
    test_data  = df_features.iloc[split_idx:][feature_cols].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    train_scaled = scaler.transform(train_data)
    test_scaled  = scaler.transform(test_data)
    
    close_idx = feature_cols.index('Close')
    
    return {
        'train_scaled': train_scaled,
        'test_scaled':  test_scaled,
        'scaler':       scaler,
        'split_idx':    split_idx,
        'close_idx':    close_idx
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
    model = Sequential([
        SimpleRNN(64, return_sequences=True, 
                 input_shape=(seq_length, n_features)),
        Dropout(0.25),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dense(1)
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

X_train, y_train = create_sequences(
    data_dict['train_scaled'], SEQ_LENGTH, data_dict['close_idx'])
X_test, y_test = create_sequences(
    data_dict['test_scaled'], SEQ_LENGTH, data_dict['close_idx'])

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
    early_stop = EarlyStopping(
        monitor='val_loss', patience=20,
        restore_best_weights=True, verbose=1)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=6, verbose=1)
    
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
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history


print("\n" + "="*70)
print("STEP 10: Training Model")
print("="*70)

history = train_model(model, X_train, y_train, X_test, y_test)

print("\n✓ Training selesai!")
print(f"  Epochs trained  : {len(history.history['loss'])}")
print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss  : {history.history['val_loss'][-1]:.6f}")
print("="*70)


# =============================================================================
# STEP 10.1 — Simpan Model & Scaler
# =============================================================================
def save_artifacts(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


print("\n" + "="*70)
print("STEP 10.1: Save Model & Scaler")
print("="*70)

saved_model_path, saved_scaler_path = save_artifacts(model, data_dict['scaler'])
print("✓ Artifact berhasil disimpan")
print(f"  Model  : {saved_model_path}")
print(f"  Scaler : {saved_scaler_path}")

print("\nContoh load ulang untuk inference:")
print("  from tensorflow import keras")
print("  import joblib")
print(f"  model  = keras.models.load_model('{MODEL_PATH}')")
print(f"  scaler = joblib.load('{SCALER_PATH}')")
print("="*70)


# =============================================================================
# STEP 11 — Evaluasi Regresi
# =============================================================================
def inverse_transform_close(scaler, values, close_idx):
    values = np.asarray(values)
    return (values - scaler.min_[close_idx]) / scaler.scale_[close_idx]


def evaluate_regression(model, X_test, y_test, scaler, close_idx):
    y_pred_scaled   = model.predict(X_test, verbose=0).flatten()
    y_actual_scaled = y_test
    
    y_actual = inverse_transform_close(scaler, y_actual_scaled, close_idx)
    y_pred   = inverse_transform_close(scaler, y_pred_scaled,   close_idx)
    
    min_len  = min(len(y_actual), len(y_pred))
    y_actual = y_actual[:min_len]
    y_pred   = y_pred[:min_len]
    
    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2   = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-8))) * 100
    
    return {'y_actual': y_actual, 'y_pred': y_pred,
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def calculate_trend_accuracy_1day(y_actual, y_pred):
    y_actual = np.asarray(y_actual).flatten()
    y_pred   = np.asarray(y_pred).flatten()
    if len(y_actual) < 2:
        raise ValueError("Minimal 2 data untuk trend 1-hari")
    actual_direction = np.sign(np.diff(y_actual))
    pred_direction   = np.sign(np.diff(y_pred))
    correct  = np.sum(actual_direction == pred_direction)
    total    = len(actual_direction)
    return {'n_days': 1, 'accuracy': (correct/total)*100,
            'correct': correct, 'total': total}


def calculate_trend_accuracy_3day(y_actual, y_pred, n_days=3):
    y_actual = np.asarray(y_actual).flatten()
    y_pred   = np.asarray(y_pred).flatten()
    if len(y_actual) <= n_days:
        raise ValueError(f"Minimal {n_days+1} data untuk trend {n_days}-hari")
    actual_cumret = (y_actual[n_days:] / (y_actual[:-n_days] + 1e-12)) - 1.0
    pred_cumret   = (y_pred[n_days:]   / (y_pred[:-n_days]   + 1e-12)) - 1.0
    actual_sign   = np.sign(actual_cumret)
    pred_sign     = np.sign(pred_cumret)
    correct = np.sum(actual_sign == pred_sign)
    total   = len(actual_sign)
    return {'n_days': n_days, 'accuracy': (correct/total)*100,
            'correct': correct, 'total': total}


print("\n" + "="*70)
print("STEP 11: Evaluasi Model Regresi")
print("="*70)

eval_results = evaluate_regression(
    model, X_test, y_test, data_dict['scaler'], data_dict['close_idx'])

print("\n📊 Metrik Regresi:")
print(f"  MAE (Mean Absolute Error)       : Rp {eval_results['MAE']:,.2f}")
print(f"  RMSE (Root Mean Squared Error)  : Rp {eval_results['RMSE']:,.2f}")
print(f"  MAPE (Mean Absolute % Error)    : {eval_results['MAPE']:.2f}%")
print(f"  R² Score (Coefficient of Det.)  : {eval_results['R2']:.4f}")

print("\n📈 Analisis Tambahan - Directional Accuracy:")

trend_1day = calculate_trend_accuracy_1day(
    eval_results['y_actual'], eval_results['y_pred'])
print(f"  Trend Accuracy 1-hari : {trend_1day['accuracy']:.2f}%")
print(f"  (Prediksi arah benar  : {trend_1day['correct']}/{trend_1day['total']})")

trend_3day = calculate_trend_accuracy_3day(
    eval_results['y_actual'], eval_results['y_pred'], n_days=3)
print(f"  Trend Accuracy 3-hari : {trend_3day['accuracy']:.2f}%")
print(f"  (Prediksi arah benar  : {trend_3day['correct']}/{trend_3day['total']})")
print(f"\n  * Derived dari hasil regresi (bukan classifier terpisah)")

print("\n📝 Sample Predictions (5 titik):")
sample_indices = np.linspace(0, len(eval_results['y_actual'])-1, 5, dtype=int)
print(f"  {'Index':<8} {'Actual (Rp)':<15} {'Predicted (Rp)':<15} {'Error (Rp)':<15}")
print("  " + "-"*55)
for idx in sample_indices:
    actual = eval_results['y_actual'][idx]
    pred   = eval_results['y_pred'][idx]
    error  = actual - pred
    print(f"  {idx:<8} {actual:>13,.0f}  {pred:>13,.0f}  {error:>13,.0f}")

print("="*70)


# =============================================================================
# STEP 12 — Visualisasi
# =============================================================================
def plot_results(history, y_actual, y_pred, mape, r2, output_file=PLOT_PATH):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(history.history['loss'],     label='Train Loss', linewidth=2)
    axes[0,0].plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
    axes[0,0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss (MSE)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(y_actual, label='Actual',    alpha=0.8, linewidth=1.5)
    axes[0,1].plot(y_pred,   label='Predicted', alpha=0.8, linewidth=1.5)
    axes[0,1].set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Harga (Rp)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].scatter(y_actual, y_pred, alpha=0.5, s=20)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    axes[1,0].plot([min_val, max_val], [min_val, max_val],
                   'r--', lw=2, label='Perfect Prediction')
    axes[1,0].set_xlabel('Actual Price (Rp)')
    axes[1,0].set_ylabel('Predicted Price (Rp)')
    axes[1,0].set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    errors = y_actual - y_pred
    axes[1,1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1,1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1,1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Error (Actual - Predicted)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'TLKM Stock Price Prediction - SimpleRNN\n'
                f'MAPE: {mape:.2f}% | R²: {r2:.4f}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return output_file


print("\n" + "="*70)
print("STEP 12: Visualisasi Hasil")
print("="*70)

output_file = plot_results(
    history,
    eval_results['y_actual'],
    eval_results['y_pred'],
    eval_results['MAPE'],
    eval_results['R2']
)

print(f"✓ Grafik berhasil disimpan: {output_file}")
print("\n  Grafik berisi:")
print("    1. Training & Validation Loss")
print("    2. Actual vs Predicted Time Series")
print("    3. Scatter Plot (Actual vs Predicted)")
print("    4. Error Distribution")
print("="*70)


# =============================================================================
# RINGKASAN AKHIR
# =============================================================================
print("\n" + "="*70)
print("🎯 RINGKASAN PROYEK")
print("="*70)
print(f"Dataset        : TLKM.JK ({df_feat.index[0].date()} - {df_feat.index[-1].date()})")
print(f"Total Samples  : {len(df_feat)}")
print(f"Features       : {len(FEATURE_COLS)} (df_feat: {df_feat.shape[1]} kolom)")
print(f"Target         : {TARGET_COL}")
print(f"Window Size    : {SEQ_LENGTH}")
print(f"Train/Test     : {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")
print(f"\nModel          : SimpleRNN (2 layers)")
print(f"Optimizer      : Adam (lr=0.0005)")
print(f"Loss Function  : MSE")
print(f"Epochs Trained : {len(history.history['loss'])}")
print(f"\nPerforma Test Set:")
print(f"  MAE          : Rp {eval_results['MAE']:,.2f}")
print(f"  RMSE         : Rp {eval_results['RMSE']:,.2f}")
print(f"  MAPE         : {eval_results['MAPE']:.2f}%")
print(f"  R²           : {eval_results['R2']:.4f}")
print(f"  Trend 1-hari : {trend_1day['accuracy']:.2f}%")
print(f"  Trend 3-hari : {trend_3day['accuracy']:.2f}%")
print(f"\nOutput Files:")
print(f"  • {output_file}")
print(f"  • {DATA_CSV_PATH}")
print(f"  • {DATA_FEAT_CSV_PATH}")
print("="*70)
print("\n✅ PROYEK SELESAI - Semua tahapan berhasil dijalankan!")
print("="*70)