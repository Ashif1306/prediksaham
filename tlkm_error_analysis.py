"""
=============================================================================
Error Analysis — Prediksi Harga Saham TLKM dengan SimpleRNN
=============================================================================
Analisis Kesalahan mencakup:
  A. Distribusi Error
  B. Identifikasi Error Terbesar
  C. Analisis Bias Model
  D. Confusion Matrix Arah Pergerakan
  E. Analisis Error Berdasarkan Volatilitas
=============================================================================
CARA PENGGUNAAN:
  MODE 1 — Lanjutan dari main script (semua variabel sudah ada di namespace):
    Cukup jalankan: python tlkm_error_analysis.py
    Variabel yang dibutuhkan dari main script:
      • eval_results : dict  — berisi y_actual, y_pred
      • df_feat      : pd.DataFrame — 11 kolom, index tanggal
      • data_dict    : dict  — berisi split_idx
      • SEQ_LENGTH   : int   — panjang sliding window (default 10)

  MODE 2 — Standalone (tanpa menjalankan main script):
    Script otomatis load file berikut yang dihasilkan main script:
      • models/tlkm_rnn_model.keras  — model yang sudah ditraining
      • data/data_tlkm_features.csv  — 11 kolom fitur bersih (BUKAN OHLCV mentah)
    Tidak perlu clean_data / create_features — data sudah siap pakai.

  Output:
    • img/tlkm_error_analysis.png  — Visualisasi 3 panel
    • Laporan teks dicetak ke konsol
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import os

# ---------------------------------------------------------------------------
# Konfigurasi Plot
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

ERROR_ANALYSIS_PLOT_PATH = "img/tlkm_error_analysis.png"
os.makedirs("img", exist_ok=True)


# =============================================================================
# FUNGSI UTAMA ERROR ANALYSIS
# =============================================================================

def run_error_analysis(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    df_feat: pd.DataFrame,
    split_idx: int,
    seq_length: int,
    plot_path: str = ERROR_ANALYSIS_PLOT_PATH,
) -> dict:
    """
    Menjalankan seluruh pipeline Error Analysis.

    Parameters
    ----------
    y_actual   : harga aktual (Rupiah), shape (n,)
    y_pred     : harga prediksi (Rupiah), shape (n,)
    df_feat    : DataFrame fitur lengkap dengan DatetimeIndex
    split_idx  : indeks pemisah train/test pada df_feat
    seq_length : panjang sliding window
    plot_path  : lokasi simpan gambar output

    Returns
    -------
    dict berisi semua hasil analisis
    """

    y_actual = np.asarray(y_actual).flatten()
    y_pred   = np.asarray(y_pred).flatten()
    n        = len(y_actual)

    # ------------------------------------------------------------------
    # Rekonstruksi index tanggal untuk test set
    # Sliding window menggeser awal seq_length langkah ke depan
    # ------------------------------------------------------------------
    df_test_full = df_feat.iloc[split_idx:].copy()
    date_index   = df_test_full.index[seq_length : seq_length + n]

    # Buat DataFrame hasil prediksi
    df_err = pd.DataFrame({
        "Date"      : date_index,
        "Actual"    : y_actual,
        "Predicted" : y_pred,
    }).set_index("Date")

    df_err["Error"]          = df_err["Actual"] - df_err["Predicted"]
    df_err["Absolute_Error"] = df_err["Error"].abs()

    # Tambahkan fitur volatilitas (tidak ada leakage — data test)
    vol_series = df_test_full["ROLL_STD_RETURN_5D"].iloc[seq_length : seq_length + n]
    vol_series.index = date_index
    df_err["Volatility"] = vol_series

    results = {}

    # ================================================================== #
    #  A. DISTRIBUSI ERROR                                                #
    # ================================================================== #
    _section("A. DISTRIBUSI ERROR")

    mean_error  = df_err["Error"].mean()
    mae         = df_err["Absolute_Error"].mean()
    std_error   = df_err["Error"].std()
    median_err  = df_err["Error"].median()
    skew_err    = df_err["Error"].skew()

    print(f"  Jumlah sampel test      : {n:,}")
    print(f"  Mean Error              : Rp {mean_error:>12,.2f}")
    print(f"  Median Error            : Rp {median_err:>12,.2f}")
    print(f"  Mean Absolute Error     : Rp {mae:>12,.2f}")
    print(f"  Std. Deviation Error    : Rp {std_error:>12,.2f}")
    print(f"  Skewness Error          : {skew_err:>12.4f}")
    print(f"  Min Error               : Rp {df_err['Error'].min():>12,.2f}")
    print(f"  Max Error               : Rp {df_err['Error'].max():>12,.2f}")

    results["dist"] = {
        "mean_error": mean_error,
        "median_error": median_err,
        "mae": mae,
        "std_error": std_error,
        "skew": skew_err,
    }

    # ================================================================== #
    #  B. IDENTIFIKASI 10 ERROR TERBESAR                                  #
    # ================================================================== #
    _section("B. IDENTIFIKASI 10 ERROR TERBESAR")

    top10 = (
        df_err[["Actual", "Predicted", "Error", "Absolute_Error"]]
        .sort_values("Absolute_Error", ascending=False)
        .head(10)
        .copy()
    )

    header = (
        f"  {'Tanggal':<12}  {'Actual (Rp)':>14}  "
        f"{'Predicted (Rp)':>14}  {'Error (Rp)':>14}  "
        f"{'|Error| (Rp)':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dt, row in top10.iterrows():
        print(
            f"  {str(dt.date()):<12}  "
            f"{row['Actual']:>14,.0f}  "
            f"{row['Predicted']:>14,.0f}  "
            f"{row['Error']:>14,.0f}  "
            f"{row['Absolute_Error']:>14,.0f}"
        )

    results["top10"] = top10

    # ================================================================== #
    #  C. ANALISIS BIAS MODEL                                             #
    # ================================================================== #
    _section("C. ANALISIS BIAS MODEL")

    n_over  = (df_err["Error"] < 0).sum()   # Predicted > Actual
    n_under = (df_err["Error"] > 0).sum()   # Predicted < Actual
    n_exact = (df_err["Error"] == 0).sum()

    pct_over  = n_over  / n * 100
    pct_under = n_under / n * 100
    pct_exact = n_exact / n * 100

    print(f"  Overestimate  (Predicted > Actual) : {n_over:>5} ({pct_over:5.1f}%)")
    print(f"  Underestimate (Predicted < Actual) : {n_under:>5} ({pct_under:5.1f}%)")
    print(f"  Tepat                              : {n_exact:>5} ({pct_exact:5.1f}%)")
    print()

    if abs(mean_error) < std_error * 0.05:
        bias_conclusion = "Model relatif TIDAK BIAS (Mean Error mendekati nol)."
    elif mean_error > 0:
        bias_conclusion = (
            f"Model cenderung UNDERESTIMATE — rata-rata memprediksi "
            f"Rp {abs(mean_error):,.0f} DI BAWAH harga aktual."
        )
    else:
        bias_conclusion = (
            f"Model cenderung OVERESTIMATE — rata-rata memprediksi "
            f"Rp {abs(mean_error):,.0f} DI ATAS harga aktual."
        )

    print(f"  ➤ Kesimpulan: {bias_conclusion}")

    results["bias"] = {
        "n_over": n_over,
        "n_under": n_under,
        "pct_over": pct_over,
        "pct_under": pct_under,
        "conclusion": bias_conclusion,
    }

    # ================================================================== #
    #  D. CONFUSION MATRIX ARAH PERGERAKAN                               #
    # ================================================================== #
    _section("D. CONFUSION MATRIX ARAH PERGERAKAN (Naik/Turun)")

    # Arah: 1 = Naik, 0 = Turun
    actual_dir = (np.diff(y_actual) > 0).astype(int)
    pred_dir   = (np.diff(y_pred)   > 0).astype(int)

    cm        = confusion_matrix(actual_dir, pred_dir)
    acc_dir   = accuracy_score(actual_dir, pred_dir)
    prec_naik = precision_score(actual_dir, pred_dir, zero_division=0)
    rec_naik  = recall_score(actual_dir, pred_dir, zero_division=0)
    f1_naik   = (
        2 * prec_naik * rec_naik / (prec_naik + rec_naik)
        if (prec_naik + rec_naik) > 0 else 0.0
    )

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix (baris=Aktual, kolom=Prediksi):\n")
    print(f"  {'':15} {'Pred Turun':>12} {'Pred Naik':>12}")
    print(f"  {'Aktual Turun':15} {tn:>12,} {fp:>12,}")
    print(f"  {'Aktual Naik':15} {fn:>12,} {tp:>12,}")
    print()
    print(f"  Accuracy  Arah       : {acc_dir*100:.2f}%")
    print(f"  Precision (Naik)     : {prec_naik*100:.2f}%")
    print(f"  Recall    (Naik)     : {rec_naik*100:.2f}%")
    print(f"  F1-Score  (Naik)     : {f1_naik*100:.2f}%")
    print()

    if acc_dir >= 0.55:
        dir_conclusion = "Model CUKUP BAIK menangkap arah pergerakan (>55%)."
    elif acc_dir >= 0.50:
        dir_conclusion = "Model SEDIKIT DI ATAS random chance (50%) dalam menangkap arah."
    else:
        dir_conclusion = "Model LEBIH BURUK dari random dalam menangkap arah pergerakan."

    print(f"  ➤ Kesimpulan: {dir_conclusion}")

    results["direction"] = {
        "cm": cm,
        "actual_dir": actual_dir,
        "pred_dir": pred_dir,
        "accuracy": acc_dir,
        "precision_naik": prec_naik,
        "recall_naik": rec_naik,
        "f1_naik": f1_naik,
        "conclusion": dir_conclusion,
    }

    # ================================================================== #
    #  E. ANALISIS ERROR BERDASARKAN VOLATILITAS                         #
    # ================================================================== #
    _section("E. ANALISIS ERROR BERDASARKAN VOLATILITAS (ROLL_STD_RETURN_5D)")

    vol_median = df_err["Volatility"].median()
    df_low_vol = df_err[df_err["Volatility"] <= vol_median]
    df_high_vol = df_err[df_err["Volatility"] > vol_median]

    mae_low  = df_low_vol["Absolute_Error"].mean()
    mae_high = df_high_vol["Absolute_Error"].mean()
    ratio    = mae_high / mae_low if mae_low > 0 else float("nan")

    print(f"  Median Volatilitas (ROLL_STD_RETURN_5D) : {vol_median:.6f}")
    print()
    print(f"  {'Kelompok':<25} {'N Sampel':>10} {'MAE (Rp)':>14}")
    print("  " + "-" * 52)
    print(f"  {'Volatilitas Rendah (≤ median)':<25} {len(df_low_vol):>10,} {mae_low:>14,.2f}")
    print(f"  {'Volatilitas Tinggi (> median)':<25} {len(df_high_vol):>10,} {mae_high:>14,.2f}")
    print(f"\n  Rasio MAE Tinggi / Rendah : {ratio:.2f}x")
    print()

    if ratio > 1.5:
        vol_conclusion = (
            f"Model JAUH LEBIH LEMAH saat volatilitas tinggi "
            f"— MAE meningkat {ratio:.1f}x lipat dibanding kondisi stabil."
        )
    elif ratio > 1.1:
        vol_conclusion = (
            f"Model SEDIKIT LEBIH LEMAH saat volatilitas tinggi "
            f"(MAE +{(ratio-1)*100:.0f}%)."
        )
    else:
        vol_conclusion = (
            "Model RELATIF KONSISTEN pada kedua kondisi volatilitas."
        )

    print(f"  ➤ Kesimpulan: {vol_conclusion}")

    results["volatility"] = {
        "vol_median": vol_median,
        "mae_low": mae_low,
        "mae_high": mae_high,
        "ratio": ratio,
        "conclusion": vol_conclusion,
        "df_low": df_low_vol,
        "df_high": df_high_vol,
    }

    # ================================================================== #
    #  VISUALISASI                                                        #
    # ================================================================== #
    _section("VISUALISASI")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Histogram Error ---
    ax = axes[0]
    ax.hist(
        df_err["Error"],
        bins=50,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
        label="Error",
    )
    ax.axvline(0,           color="red",    linestyle="--", lw=1.8, label="Zero Error")
    ax.axvline(mean_error,  color="orange", linestyle="-",  lw=1.8,
               label=f"Mean = Rp {mean_error:,.0f}")
    ax.axvline(median_err,  color="green",  linestyle=":",  lw=1.8,
               label=f"Median = Rp {median_err:,.0f}")
    ax.set_title("A. Distribusi Error\n(Actual − Predicted)")
    ax.set_xlabel("Error (Rp)")
    ax.set_ylabel("Frekuensi")
    # Formatter adaptif: pakai K jika |nilai| >= 1000, angka biasa jika lebih kecil
    def _smart_fmt(x, _):
        if abs(x) >= 1000:
            return f"{x/1000:.1f}K"
        else:
            return f"{x:,.0f}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Scatter Volatilitas vs Absolute Error ---
    ax = axes[1]
    sc = ax.scatter(
        df_err["Volatility"],
        df_err["Absolute_Error"],
        alpha=0.35,
        s=12,
        c=df_err["Volatility"],
        cmap="coolwarm",
    )
    ax.axvline(
        vol_median,
        color="navy",
        linestyle="--",
        lw=1.5,
        label=f"Median Vol = {vol_median:.5f}",
    )
    plt.colorbar(sc, ax=ax, label="Volatility")
    ax.set_title("E. Volatilitas vs Absolute Error")
    ax.set_xlabel("ROLL_STD_RETURN_5D (Volatilitas)")
    ax.set_ylabel("|Error| (Rp)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_smart_fmt))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate MAE boxes
    for grp_label, grp_mae, x_pos in [
        ("Low Vol\nMAE", mae_low, vol_median * 0.5),
        ("High Vol\nMAE", mae_high, vol_median * 1.5),
    ]:
        ax.annotate(
            f"{grp_label}\nRp {grp_mae:,.0f}",
            xy=(x_pos, df_err["Absolute_Error"].max() * 0.88),
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    # --- Panel 3: Confusion Matrix Heatmap ---
    ax = axes[2]
    cm_labels = np.array([
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ])
    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        cbar=True,
        xticklabels=["Pred Turun", "Pred Naik"],
        yticklabels=["Aktual Turun", "Aktual Naik"],
    )
    ax.set_title(
        f"D. Confusion Matrix Arah Pergerakan\n"
        f"Accuracy: {acc_dir*100:.1f}%  |  "
        f"Precision: {prec_naik*100:.1f}%  |  "
        f"Recall: {rec_naik*100:.1f}%"
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    # Suptitle
    plt.suptitle(
        "Error Analysis — Prediksi Harga Saham TLKM (SimpleRNN)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Grafik disimpan: {plot_path}")
    print("    Panel 1 : Histogram distribusi error")
    print("    Panel 2 : Scatter volatilitas vs absolute error")
    print("    Panel 3 : Confusion matrix heatmap")

    results["plot_path"]  = plot_path
    results["df_err"]     = df_err

    # ================================================================== #
    #  RINGKASAN AKHIR                                                    #
    # ================================================================== #
    _section("RINGKASAN ERROR ANALYSIS")

    print(f"  {'Metrik':<40} {'Nilai':>20}")
    print("  " + "-" * 62)
    print(f"  {'Mean Error (Bias)':40} {'Rp ' + f'{mean_error:,.0f}':>20}")
    print(f"  {'Mean Absolute Error (MAE)':40} {'Rp ' + f'{mae:,.0f}':>20}")
    print(f"  {'Std. Deviation Error':40} {'Rp ' + f'{std_error:,.0f}':>20}")
    print(f"  {'Overestimate (%)':40} {f'{pct_over:.1f}%':>20}")
    print(f"  {'Underestimate (%)':40} {f'{pct_under:.1f}%':>20}")
    print(f"  {'Directional Accuracy':40} {f'{acc_dir*100:.2f}%':>20}")
    print(f"  {'Precision Naik':40} {f'{prec_naik*100:.2f}%':>20}")
    print(f"  {'Recall Naik':40} {f'{rec_naik*100:.2f}%':>20}")
    print(f"  {'MAE (Volatilitas Rendah)':40} {'Rp ' + f'{mae_low:,.0f}':>20}")
    print(f"  {'MAE (Volatilitas Tinggi)':40} {'Rp ' + f'{mae_high:,.0f}':>20}")
    print(f"  {'Rasio MAE Tinggi/Rendah':40} {f'{ratio:.2f}x':>20}")
    print()
    print("  ── Kesimpulan ──────────────────────────────────────────────")
    print(f"  Bias     : {bias_conclusion}")
    print(f"  Arah     : {dir_conclusion}")
    print(f"  Volatil  : {vol_conclusion}")
    print("="*70)

    return results


# =============================================================================
# HELPER
# =============================================================================

def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT — dijalankan setelah main script
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Variabel di bawah ini tersedia jika script ini dijalankan           #
    # setelah / di dalam namespace main script TLKM RNN.                 #
    # Jika standalone, ganti dengan path load model & data.              #
    # ------------------------------------------------------------------ #

    try:
        # Coba akses variabel dari namespace yang sudah ada
        _y_actual   = eval_results["y_actual"]
        _y_pred     = eval_results["y_pred"]
        _df_feat    = df_feat
        _split_idx  = data_dict["split_idx"]
        _seq_length = SEQ_LENGTH

    except NameError:
        # ------------------------------------------------------------- #
        # Fallback: load artefak yang sudah disimpan oleh main script.   #
        # Main script sudah menyimpan data/data_tlkm_features.csv        #
        # (11 kolom bersih), jadi tidak perlu clean_data /               #
        # create_features lagi — cukup load langsung.                    #
        # ------------------------------------------------------------- #
        from tensorflow import keras
        from sklearn.preprocessing import MinMaxScaler

        print("Variabel runtime tidak ditemukan. Load dari file artefak...")

        # 1. Load model
        _model = keras.models.load_model("models/tlkm_rnn_model.keras")
        print("  ✓ Model loaded   : models/tlkm_rnn_model.keras")

        # 2. Load feature DataFrame — 11 kolom bersih, hasil main script
        _df_feat = pd.read_csv(
            "data/data_tlkm_features.csv",
            index_col=0, parse_dates=True
        )
        print(f"  ✓ Features loaded: data/data_tlkm_features.csv "
              f"— {len(_df_feat)} baris, {_df_feat.shape[1]} kolom")

        # 3. Parameter — harus identik dengan main script
        _FEATURE_COLS = [
            "Open", "High", "Low", "Close", "Volume",
            "RETURN_LAG_1", "RETURN_LAG_2",
            "RSI_SLOPE", "ROLL_STD_RETURN_5D",
            "MA_5", "MA_10",
        ]
        _TARGET_COL  = "Close"
        _TRAIN_RATIO = 0.8
        _SEQ_LENGTH  = 10
        _CLOSE_IDX   = _FEATURE_COLS.index(_TARGET_COL)
        _split_idx   = int(len(_df_feat) * _TRAIN_RATIO)

        # 4. Scale — fit HANYA pada train (no leakage)
        _train_arr = _df_feat.iloc[:_split_idx][_FEATURE_COLS].values
        _test_arr  = _df_feat.iloc[_split_idx:][_FEATURE_COLS].values
        _scaler    = MinMaxScaler().fit(_train_arr)
        _test_sc   = _scaler.transform(_test_arr)
        print("  ✓ Scaler fitted  : train set only (no leakage)")

        # 5. Sliding window
        def _make_seq(data, sl, ti):
            X, y = [], []
            for i in range(len(data) - sl):
                X.append(data[i:i+sl, :])
                y.append(data[i+sl, ti])
            return np.array(X), np.array(y)

        _X_test, _y_test_sc = _make_seq(_test_sc, _SEQ_LENGTH, _CLOSE_IDX)
        _y_pred_sc          = _model.predict(_X_test, verbose=0).flatten()
        print(f"  ✓ Prediksi selesai: {len(_y_pred_sc)} sampel test")

        # 6. Inverse transform ke harga Rupiah
        def _inv(scaler, vals, idx):
            return (vals - scaler.min_[idx]) / scaler.scale_[idx]

        _y_actual   = _inv(_scaler, _y_test_sc, _CLOSE_IDX)
        _y_pred     = _inv(_scaler, _y_pred_sc, _CLOSE_IDX)
        _seq_length = _SEQ_LENGTH
        print("  ✓ Inverse transform selesai — siap untuk error analysis")

    # ------------------------------------------------------------------ #
    # Jalankan analisis                                                   #
    # ------------------------------------------------------------------ #
    error_results = run_error_analysis(
        y_actual   = _y_actual,
        y_pred     = _y_pred,
        df_feat    = _df_feat,
        split_idx  = _split_idx,
        seq_length = _seq_length,
        plot_path  = ERROR_ANALYSIS_PLOT_PATH,
    )