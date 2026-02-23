"""
eda_visualization.py
====================
Exploratory Data Analysis (EDA) untuk dataset harga saham TLKM.JK.

Grafik yang dihasilkan (disimpan ke folder img/):
    1. eda_tren_close.png          — Time series harga penutupan (Close)
    2. eda_distribusi_return.png   — Histogram distribusi return harian + KDE
    3. eda_volatilitas_5d.png      — Rolling volatility 5-hari
    4. eda_heatmap_korelasi.png    — Heatmap korelasi 5 variabel data mentah (OHLCV)

Jalankan dari root folder proyek:
    python eda_visualization.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

# ── Konfigurasi global ────────────────────────────────────────────────────────
DATA_PATH  = "data/data_tlkm_harga_saham.csv"
OUTPUT_DIR = "img"
DPI        = 300

# 5 variabel data mentah untuk heatmap korelasi
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Palet warna konsisten
COLOR_CLOSE  = "#1F3864"
COLOR_RETURN = "#2E75B6"
COLOR_VOL    = "#C00000"
COLOR_KDE    = "#E67E22"
COLOR_MEAN   = "#27AE60"

sns.set(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "axes.titlesize"  : 14,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "figure.dpi"      : 100,
})


# ── 1. Load & Persiapan Data ──────────────────────────────────────────────────
def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Muat CSV, set index datetime, pastikan kolom OHLCV bertipe float,
    dan hitung fitur turunan untuk grafik tren & distribusi.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File tidak ditemukan: '{path}'\n"
            "Pastikan sudah menjalankan skrip utama agar CSV tersedia."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)

    # Flatten MultiIndex kolom jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p).strip() for p in col if str(p).strip())
            for col in df.columns.to_flat_index()
        ]

    # Rename kolom OHLCV secara fleksibel
    rename_map = {}
    for base in OHLCV_COLS:
        matches = [c for c in df.columns if c == base or c.startswith(f"{base}_")]
        if matches:
            rename_map[matches[0]] = base
    df = df.rename(columns=rename_map)

    # Index datetime
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Pilih & konversi OHLCV ke float
    available = [c for c in OHLCV_COLS if c in df.columns]
    df = df[available].apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill().dropna()
    df = df[df["Close"] > 0]

    # Fitur turunan (hanya untuk grafik tren & distribusi, BUKAN heatmap)
    df["RETURN_1D"]          = df["Close"].pct_change()
    df["ROLL_STD_RETURN_5D"] = df["RETURN_1D"].rolling(5, min_periods=5).std()
    df["MA_5"]               = df["Close"].rolling(5,  min_periods=5).mean()
    df["MA_10"]              = df["Close"].rolling(10, min_periods=10).mean()

    df = df.dropna()

    print(f"✓ Data dimuat: {len(df):,} baris trading")
    print(f"  Rentang : {df.index[0].date()} — {df.index[-1].date()}")
    print(f"  Kolom   : {list(df.columns)}\n")

    return df


# ── 2. Grafik 1 — Time Series Harga Close ────────────────────────────────────
def plot_close_trend(df: pd.DataFrame, out_dir: str) -> None:
    """Grafik time series harga penutupan harian TLKM beserta MA-5 dan MA-10."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df["Close"],
            color=COLOR_CLOSE, linewidth=1.0, alpha=0.85, label="Harga Close")
    ax.plot(df.index, df["MA_5"],
            color="#2E75B6", linewidth=1.2, linestyle="--", alpha=0.75, label="MA-5")
    ax.plot(df.index, df["MA_10"],
            color="#E67E22", linewidth=1.2, linestyle="-.", alpha=0.75, label="MA-10")

    ax.set_title("Tren Harga Penutupan Saham TLKM.JK (2008 – Sekarang)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan (Rp)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eda_tren_close.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Disimpan : {out_path}")


# ── 3. Grafik 2 — Distribusi Return Harian ───────────────────────────────────
def plot_return_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Histogram distribusi return harian + KDE + garis mean & ±1 std."""
    returns = df["RETURN_1D"].dropna()
    mean_r  = returns.mean()
    std_r   = returns.std()

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(returns, bins=80, kde=False, ax=ax,
                 color=COLOR_RETURN, stat="density", alpha=0.55,
                 label="Return Harian")
    sns.kdeplot(returns, ax=ax,
                color=COLOR_KDE, linewidth=2, label="KDE")

    ax.axvline(mean_r,         color=COLOR_MEAN, linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_r:.4f}")
    ax.axvline(mean_r + std_r, color=COLOR_VOL,  linestyle=":",  linewidth=1.5,
               label=f"+1σ = {mean_r + std_r:.4f}")
    ax.axvline(mean_r - std_r, color=COLOR_VOL,  linestyle=":",  linewidth=1.5,
               label=f"−1σ = {mean_r - std_r:.4f}")

    stats_text = (
        f"n     = {len(returns):,}\n"
        f"μ     = {mean_r:.4f}\n"
        f"σ     = {std_r:.4f}\n"
        f"Skew  = {returns.skew():.3f}\n"
        f"Kurt  = {returns.kurtosis():.3f}"
    )
    ax.text(0.97, 0.95, stats_text,
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#BBBBBB", alpha=0.85))

    ax.set_title("Distribusi Return Harian Saham TLKM.JK")
    ax.set_xlabel("Return Harian (pct_change)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eda_distribusi_return.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Disimpan : {out_path}")


# ── 4. Grafik 3 — Rolling Volatility 5-Hari ──────────────────────────────────
def plot_rolling_volatility(df: pd.DataFrame, out_dir: str) -> None:
    """Grafik rolling volatility 5-hari (std return) dengan area shading."""
    vol = df["ROLL_STD_RETURN_5D"].dropna()

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(vol.index, vol,
            color=COLOR_VOL, linewidth=0.9, alpha=0.85, label="Volatilitas 5-hari")
    ax.fill_between(vol.index, vol, alpha=0.18, color=COLOR_VOL)

    median_vol = vol.median()
    ax.axhline(median_vol, color=COLOR_CLOSE, linestyle="--", linewidth=1.3,
               label=f"Median = {median_vol:.5f}")

    ax.set_title("Rolling Volatility 5-Hari — Saham TLKM.JK (Std Return Harian)")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Standar Deviasi Return")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eda_volatilitas_5d.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Disimpan : {out_path}")


# ── 5. Grafik 4 — Heatmap Korelasi 5 Variabel Data Mentah ───────────────────
def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Heatmap korelasi Pearson untuk 5 variabel data mentah:
    Open, High, Low, Close, Volume.
    """
    # Ambil HANYA 5 kolom OHLCV mentah — tanpa fitur turunan apapun
    corr_df = df[OHLCV_COLS].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=0.8, linecolor="#DDDDDD",
        square=True, ax=ax,
        xticklabels=OHLCV_COLS,
        yticklabels=OHLCV_COLS,
        cbar_kws={"shrink": 0.75, "label": "Korelasi Pearson"},
    )

    ax.set_title(
        "Heatmap Korelasi — 5 Variabel Data Mentah OHLCV (TLKM.JK)",
        pad=15,
    )
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eda_heatmap_korelasi.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Disimpan : {out_path}")


# ── 6. main() ────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  EDA Visualization — TLKM.JK Stock Price Dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] Memuat dan menyiapkan data...")
    df = load_and_prepare(DATA_PATH)

    print("[2/5] Membuat grafik tren harga Close...")
    plot_close_trend(df, OUTPUT_DIR)

    print("[3/5] Membuat grafik distribusi return harian...")
    plot_return_distribution(df, OUTPUT_DIR)

    print("[4/5] Membuat grafik rolling volatility 5-hari...")
    plot_rolling_volatility(df, OUTPUT_DIR)

    print("[5/5] Membuat heatmap korelasi 5 variabel data mentah (OHLCV)...")
    plot_correlation_heatmap(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ✅ Semua grafik EDA berhasil dibuat!")
    print(f"  📁 Lokasi output : ./{OUTPUT_DIR}/")
    print("=" * 60)
    print("\nFile yang dihasilkan:")
    for fname in [
        "eda_tren_close.png",
        "eda_distribusi_return.png",
        "eda_volatilitas_5d.png",
        "eda_heatmap_korelasi.png",
    ]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
        print(f"  • {fpath:<45}  ({size:,.1f} KB)")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()