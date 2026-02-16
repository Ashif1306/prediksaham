"""
Prediksi harga saham TLKM berbasis regresi menggunakan SimpleRNN.
Fokus: pipeline regresi + analisis Trend Accuracy 3-hari (derived direction).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 10
TRAIN_RATIO = 0.8
LEARNING_RATE = 0.0005
TARGET_COL = "Close"


def load_data(csv_path: str = "data_tlkm_harga_saham.csv") -> pd.DataFrame:
    """Load data saham dari CSV lokal."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {csv_path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning data OHLCV agar siap untuk feature engineering."""
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(p).strip() for p in col if str(p).strip()])
            for col in data.columns.to_flat_index()
        ]

    required = ["Open", "High", "Low", "Close", "Volume"]
    col_map = {}
    for base in required:
        candidates = [c for c in data.columns if c == base or c.startswith(f"{base}_")]
        if candidates:
            col_map[base] = candidates[0]

    if len(col_map) < 5:
        raise ValueError(f"Kolom OHLCV tidak lengkap. Kolom tersedia: {list(data.columns)}")

    data = data[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()]
    data = data[~data.index.duplicated(keep="first")]

    data = data.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    data = data.ffill().bfill().dropna()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data = data[data[col] > 0]

    return data.sort_index()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering final untuk tugas regresi."""
    data = df.copy()

    data["SMA_10"] = data["Close"].rolling(window=10, min_periods=10).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()

    data["RETURN_1D"] = data["Close"].pct_change()
    data["RETURN_LAG_1"] = data["RETURN_1D"].shift(1)
    data["RETURN_LAG_2"] = data["RETURN_1D"].shift(2)
    data["RSI_SLOPE"] = data["RSI_14"].diff()
    data["ROLL_STD_RETURN_5D"] = data["RETURN_1D"].rolling(window=5, min_periods=5).std()

    data = data.dropna().copy()

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "EMA_10",
        "RSI_14",
        "MACD",
        "MACD_SIGNAL",
        "RETURN_LAG_1",
        "RETURN_LAG_2",
        "RSI_SLOPE",
        "ROLL_STD_RETURN_5D",
    ]

    return data[feature_cols]


def run_basic_eda(df: pd.DataFrame):
    """Ringkasan EDA dasar untuk memantau kualitas data time series."""
    print(f"Shape data        : {df.shape}")
    print(f"Rentang tanggal   : {df.index.min().date()} s.d. {df.index.max().date()}")
    print(f"Missing values    : {int(df.isna().sum().sum())}")
    print("Statistik Close   :")
    print(df["Close"].describe()[["min", "max", "mean", "std"]])


def scale_data(data: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """Split 80:20 tanpa shuffle dan fit MinMaxScaler hanya pada train set."""
    split_idx = int(len(data) * train_ratio)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    return train_scaled, test_scaled, scaler, split_idx


def create_sequences(data: np.ndarray, window_size: int = WINDOW_SIZE, target_idx: int = 3):
    """Buat sliding window untuk supervised time series."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def build_model(window_size: int, n_features: int, learning_rate: float = LEARNING_RATE) -> Sequential:
    """Arsitektur SimpleRNN untuk regresi harga."""
    model = Sequential(
        [
            SimpleRNN(64, return_sequences=True, input_shape=(window_size, n_features)),
            Dropout(0.2),
            SimpleRNN(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray):
    """Training model dengan EarlyStopping dan ReduceLROnPlateau."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )
    return history


def inverse_close(scaled_close: np.ndarray, scaler: MinMaxScaler, close_idx: int = 3) -> np.ndarray:
    """Inverse transform khusus kolom Close dari MinMaxScaler."""
    scaled_close = np.asarray(scaled_close).reshape(-1)
    return (scaled_close - scaler.min_[close_idx]) / scaler.scale_[close_idx]


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung metrik regresi: MAE, RMSE, MAPE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def calculate_trend_accuracy_3day(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Trend Accuracy 3-hari dari arah return kumulatif 3-hari (derived direction)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("Panjang y_true dan y_pred harus sama.")
    if len(y_true) < 4:
        raise ValueError("Minimal 4 titik data untuk Trend Accuracy 3-hari.")

    actual_ret_3d = (y_true[3:] / (y_true[:-3] + 1e-8)) - 1.0
    pred_ret_3d = (y_pred[3:] / (y_pred[:-3] + 1e-8)) - 1.0

    actual_dir = np.sign(actual_ret_3d)
    pred_dir = np.sign(pred_ret_3d)

    return float(np.mean(actual_dir == pred_dir) * 100)


def plot_results(index_test: pd.Index, y_true: np.ndarray, y_pred: np.ndarray, output_path: str):
    """Plot Actual vs Predicted."""
    plt.figure(figsize=(14, 6))
    plt.plot(index_test, y_true, label="Actual Close", linewidth=2)
    plt.plot(index_test, y_pred, label="Predicted Close", linewidth=2)
    plt.title("SimpleRNN Regression - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
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
