# Prediksi Harga Saham PT Telkom Indonesia (TLKM.JK) Menggunakan Recurrent Neural Network (RNN)

## 1. Deskripsi Project
Project ini merupakan sistem **Machine Learning end-to-end** untuk memprediksi harga penutupan saham **PT Telkom Indonesia (TLKM.JK)** pada **hari perdagangan berikutnya (next trading day)** menggunakan arsitektur **SimpleRNN (TensorFlow/Keras)**.

Aplikasi dilengkapi dengan:
- **REST API berbasis FastAPI** untuk inferensi, histori, dan data chart.
- **Dashboard web interaktif** (HTML/CSS/JavaScript + ApexCharts) untuk visualisasi prediksi dan pergerakan harga.
- **Penyimpanan histori prediksi** ke database **SQLite**.
- **Auto update data** saham dari **Yahoo Finance (yfinance)** sebelum proses prediksi dijalankan.

Dengan struktur ini, pengguna baru dapat memulai dari nol: instalasi, training model, menjalankan API, membuka dashboard, hingga melakukan prediksi.

---

## 2. Fitur Utama

- ✅ **Prediksi Next Trading Day**
  - Endpoint `/predict` menjalankan pipeline inferensi untuk memprediksi harga penutupan TLKM pada hari bursa berikutnya.
- ✅ **Dashboard Interaktif**
  - Tampilan modern untuk menjalankan prediksi, melihat histori, dan chart harga.
- ✅ **Chart Multi-Range**
  - Pilihan rentang data: **1D, 5D, 1M, YTD, 1Y, 3Y**.
- ✅ **Auto Update Data**
  - Data CSV historis akan diperbarui otomatis dari Yahoo Finance saat prediksi dipanggil.
- ✅ **Riwayat Prediksi ke SQLite**
  - Hasil prediksi disimpan ke tabel `prediction_history` pada file `predictions.db`.

---

## 3. Arsitektur Sistem

### 3.1 Backend (FastAPI)
- Menyediakan endpoint utama:
  - `/predict`
  - `/history`
  - `/chart-data`
- Bertanggung jawab untuk:
  - Load model dan scaler saat startup.
  - Menjalankan update data terbaru.
  - Menyimpan dan membaca histori prediksi dari SQLite.

### 3.2 Model Machine Learning (RNN)
- Model: **SimpleRNN** (TensorFlow/Keras).
- Input: window deret waktu multivariat (fitur OHLCV + fitur teknikal).
- Output: prediksi nilai `Close` untuk next trading day.

### 3.3 Database (SQLite)
- Database lokal ringan: `predictions.db`.
- Menyimpan histori prediksi dan metadata (tanggal prediksi, perubahan harga, tren, timestamp).

### 3.4 Frontend (HTML/CSS/JS)
- File template: `templates/index.html`.
- Fitur utama UI:
  - Trigger prediksi.
  - Tabel histori prediksi.
  - Chart candlestick/volume multi-range.

---

## 4. Struktur Folder & Penjelasan File

```bash
prediksaham/
├── api.py
├── predict_next_day.py
├── tlkm_stock_prediction_rnn.py
├── requirements.txt
├── predictions.db
├── data/
│   ├── data_tlkm_harga_saham.csv
│   └── data_tlkm_features.csv
├── models/
│   ├── tlkm_rnn_model.keras
│   └── tlkm_scaler.pkl
├── templates/
│   └── index.html
├── img/
│   ├── tlkm_rnn_results.png
│   ├── tlkm_error_analysis.png
│   └── ...
└── predictions/
    └── latest_prediction.json
```

### Penjelasan komponen:
- `api.py`  
  Aplikasi FastAPI untuk endpoint prediksi, histori, chart data, status, dan rendering dashboard.

- `predict_next_day.py`  
  Script inferensi mandiri untuk memprediksi hari trading berikutnya menggunakan model yang sudah dilatih.

- `tlkm_stock_prediction_rnn.py`  
  Script utama training model RNN: unduh data, preprocessing, feature engineering, training, evaluasi, dan simpan model/scaler.

- `requirements.txt`  
  Daftar dependency Python.

- `data/`  
  Menyimpan dataset historis mentah dan hasil feature engineering.

- `models/`  
  Menyimpan artefak model terlatih (`.keras`) dan scaler (`.pkl`).

- `templates/`  
  Frontend dashboard.

- `img/`  
  Visualisasi hasil EDA/training/evaluasi.

- `predictions/`  
  Menyimpan output prediksi terbaru dalam format JSON.

- `predictions.db`  
  Database SQLite untuk histori prediksi.

---

## 5. Persyaratan Sistem

- **OS**: Windows / Linux / macOS
- **Python**: disarankan **3.11 atau 3.12** (TensorFlow belum mendukung Python 3.14)
- **pip**: versi terbaru
- **Virtual environment**: `venv` (direkomendasikan)
- **Koneksi internet**: diperlukan untuk mengambil data dari Yahoo Finance

---

## 6. Langkah Instalasi dari Awal

## 6.1 Clone Repository
```bash
git clone <URL_REPOSITORY_ANDA>
cd prediksaham
```

## 6.2 Buat Virtual Environment
### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 6.3 Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6.4 Setup Awal (Opsional)
Jika menggunakan Windows dan ingin setup otomatis yang sudah disediakan:
```bat
setup.bat
```

---

## 7. Cara Training Model (Jika Model Belum Ada)
Jalankan script training berikut untuk menghasilkan model dan scaler:

```bash
python tlkm_stock_prediction_rnn.py
```

Output yang diharapkan:
- `models/tlkm_rnn_model.keras`
- `models/tlkm_scaler.pkl`
- `data/data_tlkm_harga_saham.csv`
- `data/data_tlkm_features.csv`
- visualisasi hasil training pada folder `img/`

> Jika file model dan scaler sudah ada di folder `models/`, Anda bisa langsung menjalankan API tanpa training ulang.

---

## 8. Cara Menjalankan API Server

```bash
uvicorn api:app --reload
```

Server default berjalan di:
- `http://127.0.0.1:8000`

Dokumentasi interaktif (Swagger):
- `http://127.0.0.1:8000/docs`

---

## 9. Cara Mengakses Dashboard di Browser
Setelah server aktif, buka:

```text
http://127.0.0.1:8000
```

Dashboard akan menampilkan:
- chart harga TLKM multi-range,
- tombol jalankan prediksi,
- tabel histori prediksi,
- tabel harga terbaru.

---

## 10. Penjelasan Endpoint API

## 10.1 `GET /predict`
Menjalankan prediksi harga penutupan untuk hari trading berikutnya.

Proses internal ringkas:
1. Update data terbaru dari Yahoo Finance.
2. Load data + feature engineering.
3. Inference model RNN.
4. Simpan/Update hasil ke SQLite.

## 10.2 `GET /history`
Mengambil histori prediksi dari database SQLite (default limit 100 data terbaru).

Contoh:
```bash
curl "http://127.0.0.1:8000/history?limit=20"
```

## 10.3 `GET /chart-data`
Mengambil data OHLCV untuk chart dashboard.

Parameter:
- `range`: `1D | 5D | 1M | YTD | 1Y | 3Y`

Contoh:
```bash
curl "http://127.0.0.1:8000/chart-data?range=1M"
```

---

## 11. Cara Update Data Otomatis
- Auto update data terjadi saat endpoint `/predict` dipanggil.
- API akan membandingkan tanggal terakhir pada file `data/data_tlkm_harga_saham.csv`, kemudian menarik data terbaru dari Yahoo Finance jika tersedia.

Untuk update + prediksi secara manual (tanpa dashboard), jalankan:
```bash
curl "http://127.0.0.1:8000/predict"
```

---

## 12. Cara Menggunakan Fitur Dashboard

1. Jalankan API server.
2. Buka `http://127.0.0.1:8000`.
3. Pilih range chart (`1D/5D/1M/YTD/1Y/3Y`) untuk melihat pergerakan harga.
4. Klik tombol **Jalankan Prediksi** untuk inferensi next trading day.
5. Lihat hasil prediksi dan perubahan harga.
6. Lihat histori prediksi yang tersimpan di database.

---

## 13. Contoh Output JSON Endpoint

### 13.1 Contoh `GET /predict`
```json
{
  "predicted_price": 3492.17,
  "prediction_date": "2026-02-20",
  "prediction_day": "Friday",
  "last_actual_date": "2026-02-19",
  "last_actual_day": "Thursday",
  "last_actual_price": 3480.0,
  "price_change": 12.17,
  "price_change_pct": 0.3496,
  "trend": "bullish",
  "timestamp": "2026-02-24 06:02:57",
  "saved_id": 123,
  "is_new_record": true
}
```

### 13.2 Contoh `GET /history`
```json
{
  "total": 2,
  "history": [
    {
      "id": 2,
      "prediction_date": "2026-02-20",
      "predicted_price": 3492.17,
      "last_actual_price": 3480.0,
      "price_change": 12.17,
      "price_change_pct": 0.3496,
      "trend": "bullish",
      "timestamp": "2026-02-24 06:02:57"
    }
  ]
}
```

### 13.3 Contoh `GET /chart-data?range=1M`
```json
{
  "range": "1M",
  "ticker": "TLKM.JK",
  "labels": ["1 Feb", "2 Feb", "3 Feb"],
  "candles": [
    {"o": 3500.0, "h": 3520.0, "l": 3480.0, "c": 3490.0}
  ],
  "volumes": [
    {"y": 12345678, "color": "#0af5b0"}
  ],
  "latest": {
    "close": 3490.0,
    "change": 10.0,
    "change_pct": 0.29,
    "high": 3560.0,
    "low": 3400.0,
    "date": "3 Feb"
  }
}
```

---

## 14. Penjelasan Database `predictions.db`

Database SQLite ini digunakan untuk menyimpan histori inferensi.

Tabel utama: `prediction_history`

Kolom penting:
- `id` (INTEGER, PK, AUTOINCREMENT)
- `prediction_date` (TEXT, UNIQUE)
- `predicted_price` (REAL)
- `last_actual_price` (REAL)
- `price_change` (REAL)
- `price_change_pct` (REAL)
- `trend` (TEXT: bullish/bearish/flat)
- `timestamp` (TEXT)

Catatan:
- Jika prediksi untuk `prediction_date` yang sama dijalankan ulang, record akan di-**update** (upsert), bukan duplikasi.

---

## 15. Troubleshooting Umum

### 15.1 Error: model/scaler tidak ditemukan
**Gejala:**
- `Model tidak ditemukan: models/tlkm_rnn_model.keras`
- `Scaler tidak ditemukan: models/tlkm_scaler.pkl`

**Solusi:**
1. Jalankan training: `python tlkm_stock_prediction_rnn.py`
2. Pastikan file model dan scaler muncul di folder `models/`.

### 15.2 Error yfinance / gagal download data
**Kemungkinan penyebab:**
- Koneksi internet bermasalah
- Rate limit/isu sementara Yahoo Finance

**Solusi:**
- Coba ulang beberapa saat lagi.
- Pastikan internet aktif.
- Pastikan dependensi `yfinance` terpasang dengan benar.

### 15.3 Error TensorFlow di Python 3.14
**Penyebab:**
- Versi Python tidak kompatibel.

**Solusi:**
- Gunakan Python 3.11 atau 3.12.
- Buat ulang virtual environment, lalu install ulang dependency.

### 15.4 Dashboard tidak tampil sempurna
**Solusi cepat:**
- Pastikan API aktif di `127.0.0.1:8000`.
- Hard refresh browser (Ctrl+F5).
- Cek console browser dan log terminal API.

---

## 16. Rencana Pengembangan

Beberapa pengembangan yang direkomendasikan:
- Menambahkan endpoint evaluasi metrik model real-time.
- Menambahkan scheduler (misal APScheduler/Cron) untuk update data terjadwal otomatis.
- Menyediakan fitur backtesting strategi sederhana.
- Menambahkan autentikasi dan role untuk multi-user.
- Menambahkan CI/CD pipeline dan testing otomatis.
- Eksplorasi model lanjutan (LSTM/GRU/Transformer) sebagai pembanding performa.

---

## 17. License

Project ini dapat menggunakan lisensi **MIT** (opsional). Jika diperlukan, tambahkan file `LICENSE` terpisah.

---

## 18. Ringkasan Quick Start

```bash
# 1) Clone
git clone <URL_REPOSITORY_ANDA>
cd prediksaham

# 2) Buat & aktifkan venv
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
# .\venv\Scripts\activate

# 3) Install dependency
pip install -r requirements.txt

# 4) (Opsional) Training model jika belum ada
python tlkm_stock_prediction_rnn.py

# 5) Jalankan API
uvicorn api:app --reload

# 6) Buka dashboard
# http://127.0.0.1:8000
```
