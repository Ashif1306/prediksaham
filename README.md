Prediksi Harga Saham PT Telkom Indonesia (TLKM.JK)
Menggunakan Recurrent Neural Network (RNN)

==================================================

DESKRIPSI PROJECT

Project ini merupakan sistem Machine Learning end-to-end untuk memprediksi harga penutupan saham PT Telkom Indonesia (TLKM.JK) pada hari perdagangan berikutnya menggunakan arsitektur SimpleRNN (TensorFlow/Keras).

Sistem ini dilengkapi dengan REST API berbasis FastAPI, dashboard web interaktif (HTML/CSS/JS + ApexCharts), penyimpanan histori prediksi ke SQLite, serta auto-update data saham dari Yahoo Finance.

==================================================

FITUR UTAMA

- Prediksi Next Trading Day
  Endpoint /predict untuk memprediksi harga penutupan TLKM hari bursa berikutnya.

- Dashboard Interaktif
  Tampilan modern untuk menjalankan prediksi, histori, dan chart harga.

- Chart Multi-Range
  Rentang data: 1D, 5D, 1M, YTD, 1Y, 3Y.

- Auto Update Data
  Data CSV historis diperbarui otomatis dari Yahoo Finance saat prediksi dipanggil.

- Histori ke SQLite
  Hasil prediksi disimpan ke tabel prediction_history pada predictions.db.

==================================================

ARSITEKTUR SISTEM

  Frontend (HTML)  <-->  FastAPI (api.py)  <-->  SQLite DB
  ApexCharts/JS          /predict                predictions.db
                         /history
                         /chart-data
                              |
                         Model RNN (Keras)
                         + Scaler (pkl)
                              |
                         Yahoo Finance (yfinance)

Komponen utama:

- Backend (FastAPI)
  Menyajikan endpoint prediksi, histori, dan chart; bertanggung jawab load model/scaler, update data, dan manajemen SQLite.

- Model ML (SimpleRNN)
  Input: window deret waktu multivariat (OHLCV + fitur teknikal).
  Output: prediksi nilai Close next trading day.

- Database (SQLite)
  Penyimpanan lokal histori prediksi beserta metadata.

- Frontend (HTML/JS)
  Dashboard interaktif dengan trigger prediksi, tabel histori, dan candlestick chart.

==================================================

STRUKTUR FOLDER

prediksaham/
├── api.py                          -> Aplikasi FastAPI (endpoint + dashboard)
├── predict_next_day.py             -> Script inferensi mandiri
├── tlkm_stock_prediction_rnn.py    -> Script training model RNN
├── requirements.txt                -> Dependency Python
├── predictions.db                  -> Database SQLite histori prediksi
├── data/
│   ├── data_tlkm_harga_saham.csv  -> Dataset historis harga saham
│   └── data_tlkm_features.csv     -> Dataset hasil feature engineering
├── models/
│   ├── tlkm_rnn_model.keras       -> Model terlatih
│   └── tlkm_scaler.pkl            -> Scaler tersimpan
├── templates/
│   └── index.html                 -> Frontend dashboard
├── img/                           -> Visualisasi hasil EDA/training/evaluasi
└── predictions/
    └── latest_prediction.json     -> Output prediksi terbaru (JSON)

==================================================

PERSYARATAN SISTEM

- OS       : Windows / Linux / macOS
- Python   : 3.11 atau 3.12 (TensorFlow belum mendukung Python 3.13+)
- pip      : versi terbaru
- Venv     : venv (direkomendasikan)
- Internet : diperlukan untuk mengambil data dari Yahoo Finance

==================================================

QUICK START

  1. Clone repository
     git clone <URL_REPOSITORY_ANDA>
     cd prediksaham

  2. Buat dan aktifkan virtual environment
     python -m venv venv
     source venv/bin/activate          (Linux/macOS)
     .\venv\Scripts\Activate.ps1       (Windows PowerShell)

  3. Install dependency
     pip install --upgrade pip
     pip install -r requirements.txt

  4. Training model (lewati jika model sudah ada di folder models/)
     python tlkm_stock_prediction_rnn.py

  5. Jalankan API server
     uvicorn api:app --reload

  6. Buka dashboard di browser
     http://127.0.0.1:8000

  Catatan: Pengguna Windows bisa menjalankan setup.bat untuk setup otomatis (langkah 2-3).

==================================================

TRAINING MODEL

Jalankan script berikut untuk menghasilkan model dan scaler dari awal:

  python tlkm_stock_prediction_rnn.py

Output yang dihasilkan:
  models/tlkm_rnn_model.keras
  models/tlkm_scaler.pkl
  data/data_tlkm_harga_saham.csv
  data/data_tlkm_features.csv
  img/  (visualisasi hasil training)

Jika file model dan scaler sudah ada di folder models/, langkah ini bisa dilewati.

==================================================

MENJALANKAN API SERVER

  uvicorn api:app --reload

  URL                          Deskripsi
  http://127.0.0.1:8000        Dashboard web
  http://127.0.0.1:8000/docs   Dokumentasi interaktif (Swagger UI)

==================================================

ENDPOINT API

--------------------------------------------------
GET /predict
--------------------------------------------------
Menjalankan prediksi harga penutupan untuk hari trading berikutnya.

Proses internal:
  1. Update data terbaru dari Yahoo Finance
  2. Load data + feature engineering
  3. Inference model RNN
  4. Simpan/update hasil ke SQLite

Contoh response:
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

--------------------------------------------------
GET /history
--------------------------------------------------
Mengambil histori prediksi dari database SQLite.
Parameter: limit (default: 100)

  curl "http://127.0.0.1:8000/history?limit=20"

Contoh response:
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

--------------------------------------------------
GET /chart-data
--------------------------------------------------
Mengambil data OHLCV untuk chart dashboard.
Parameter: range -> 1D | 5D | 1M | YTD | 1Y | 3Y

  curl "http://127.0.0.1:8000/chart-data?range=1M"

Contoh response:
  {
    "range": "1M",
    "ticker": "TLKM.JK",
    "labels": ["1 Feb", "2 Feb", "3 Feb"],
    "candles": [
      { "o": 3500.0, "h": 3520.0, "l": 3480.0, "c": 3490.0 }
    ],
    "volumes": [
      { "y": 12345678, "color": "#0af5b0" }
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

==================================================

DATABASE predictions.db

Tabel: prediction_history

  Kolom               Tipe          Keterangan
  id                  INTEGER       Primary key, auto-increment
  prediction_date     TEXT (UNIQUE) Tanggal prediksi
  predicted_price     REAL          Harga prediksi
  last_actual_price   REAL          Harga aktual terakhir
  price_change        REAL          Selisih harga
  price_change_pct    REAL          Persentase perubahan
  trend               TEXT          bullish / bearish / flat
  timestamp           TEXT          Waktu inferensi dijalankan

Catatan: Jika prediksi untuk tanggal yang sama dijalankan ulang, record akan di-update (upsert), bukan diduplikasi.

==================================================

CARA MENGGUNAKAN DASHBOARD

  1. Jalankan API server (uvicorn api:app --reload)
  2. Buka http://127.0.0.1:8000 di browser
  3. Pilih range chart (1D / 5D / 1M / YTD / 1Y / 3Y)
  4. Klik tombol Jalankan Prediksi untuk inferensi next trading day
  5. Lihat hasil prediksi, perubahan harga, dan tren
  6. Scroll ke bawah untuk melihat tabel histori prediksi

==================================================

TROUBLESHOOTING

Model/scaler tidak ditemukan
  Error: "Model tidak ditemukan: models/tlkm_rnn_model.keras"
  Solusi: Jalankan training terlebih dahulu.
    python tlkm_stock_prediction_rnn.py

Gagal download data dari Yahoo Finance
  Penyebab: Koneksi internet bermasalah atau rate limit Yahoo Finance.
  Solusi: Tunggu beberapa saat dan coba lagi. Pastikan yfinance terpasang dengan benar.

Error TensorFlow di Python 3.13+
  Penyebab: TensorFlow belum mendukung versi Python tersebut.
  Solusi: Gunakan Python 3.11 atau 3.12, buat ulang virtual environment, lalu install ulang dependency.

Dashboard tidak tampil sempurna
  Solusi:
  - Pastikan API aktif di http://127.0.0.1:8000
  - Hard refresh browser: Ctrl + F5
  - Periksa console browser dan log terminal API

==================================================

RENCANA PENGEMBANGAN

  - Endpoint evaluasi metrik model secara real-time
  - Scheduler otomatis (APScheduler / Cron) untuk update data terjadwal
  - Fitur backtesting strategi sederhana
  - Autentikasi dan role management untuk multi-user
  - CI/CD pipeline dan automated testing
  - Eksplorasi model lanjutan: LSTM, GRU, Transformer

==================================================

LICENSE

Project ini dilisensikan di bawah MIT License. Lihat file LICENSE untuk detail lebih lanjut.