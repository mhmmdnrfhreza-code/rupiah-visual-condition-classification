# Rupiah Visual Condition Classification

Aplikasi Web Flask Untuk Mengidentifikasi Kondisi Visual Uang Kertas Rupiah Menggunakan Ekstraksi Fitur Gray Level Co-occurrence Matrix (GLCM) dan Model Random Forest.

## Fitur

- Upload gambar uang kertas Rupiah.
- Preprocessing gambar: resize 256 x 256 dan konversi grayscale.
- Ekstraksi 6 fitur GLCM rata-rata: contrast, dissimilarity, homogeneity, energy, correlation, ASM.
- Klasifikasi kondisi visual: `normal`, `dirty`, `scuffed`, `scuffed-dirty`, `torn`.
- Konversi prediksi menjadi status kelayakan visual.
- Halaman hasil prediksi dengan preview, confidence, dan tabel fitur GLCM.
- Halaman Model berisi ringkasan model, fitur GLCM, distribusi dataset, metrik evaluasi, confusion matrix, contoh visual kelas, dan keterbatasan model.
- Script training yang menghasilkan model, fitur CSV, metrics, classification report, dan confusion matrix.

## Tech Stack

- Python
- Flask
- OpenCV
- scikit-image
- scikit-learn
- pandas
- Matplotlib dan Seaborn
- HTML, CSS, Bootstrap CDN, JavaScript sederhana

## Struktur Project

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- dataset/
|-- artifacts/
|   |-- model.pkl
|   |-- label_encoder.pkl
|   |-- features.csv
|   |-- metrics.json
|   |-- classification_report.txt
|   `-- confusion_matrix.png
|-- src/
|   |-- config.py
|   |-- dataset_loader.py
|   |-- glcm_features.py
|   |-- model_page.py
|   |-- predict_model.py
|   |-- preprocessing.py
|   |-- train_model.py
|   `-- utils.py
|-- static/
|   |-- css/style.css
|   |-- examples/
|   |-- images/confusion_matrix.png
|   |-- js/script.js
|   `-- uploads/
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- result.html
|   |-- model.html
|   `-- about.html
`-- tests/
    |-- test_app.py
    `-- test_core.py
```

## Dataset

Dataset berada di folder `dataset/` dengan struktur:

```text
dataset/
|-- 1000/
|   |-- dirty/
|   |-- normal/
|   |-- scuffed/
|   |-- scuffed-dirty/
|   `-- torn/
|-- 2000/
|-- 5000/
|-- 10000/
|-- 20000/
|-- 50000/
`-- 100000/
```

Label prediksi diambil dari nama folder kondisi, bukan dari nama file gambar. Dataset lokal saat ini berisi 350 gambar PNG: 7 nominal, 5 kondisi, dan 10 gambar per kombinasi nominal-kondisi. Ukuran folder dataset sekitar 122.79 MB, jadi Anda dapat memutuskan sendiri apakah dataset ikut diunggah ke GitHub.

## Instalasi

Gunakan Python 3.11 atau 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika perintah `python` tidak tersedia di Windows, coba Python Launcher:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Environment Variable

| Name | Required | Description |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | Tidak untuk lokal | Secret key Flask untuk flash message/session. Jika kosong, app memakai nilai default lokal `rvcc-local-dev`. |
| `FLASK_DEBUG` | Tidak | Set `1` jika ingin menjalankan Flask debug mode saat pengembangan lokal. |

## Training Model

Jalankan training dari root project:

```bash
python -m src.train_model --dataset dataset --artifacts artifacts
```

Output training:

- `artifacts/model.pkl`
- `artifacts/label_encoder.pkl`
- `artifacts/features.csv`
- `artifacts/metrics.json`
- `artifacts/classification_report.txt`
- `artifacts/confusion_matrix.png`

Jika model di-training ulang dan ingin page Model menampilkan confusion matrix terbaru, salin file hasil training:

```bash
copy artifacts\confusion_matrix.png static\images\confusion_matrix.png
```

## Menjalankan Aplikasi

Setelah model tersedia:

```bash
python app.py
```

Buka:

```text
http://127.0.0.1:5000
```

Halaman yang tersedia:

- `http://127.0.0.1:5000/` untuk upload dan prediksi.
- `http://127.0.0.1:5000/model` untuk informasi model dan evaluasi.
- `http://127.0.0.1:5000/about` untuk penjelasan metode dan batasan proyek.

## Testing

```bash
python -m unittest discover -s tests
```

## File yang Tidak Perlu Diupload

File/folder berikut sudah masuk `.gitignore`:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `*.log`
- `*.tmp`
- `.env`
- `.vscode/`
- `.idea/`
- `static/uploads/*` kecuali `.gitkeep`

## Catatan Keamanan

- Jangan jalankan `model.pkl` atau `label_encoder.pkl` dari sumber tidak tepercaya. File tersebut dibaca dengan `joblib`, jadi anggap sebagai artifact lokal hasil training sendiri.
- Jangan commit `.env`, token, credential, atau file upload user.
- Aplikasi ini ditujukan untuk berjalan lokal. Jika ingin deploy publik, gunakan secret key kuat dan WSGI server produksi.

## Catatan Keterbatasan

- Model menggunakan fitur tekstur GLCM, bukan CNN atau deep learning.
- Prediksi dapat kurang stabil jika gambar upload memiliki pencahayaan, latar, atau sudut yang sangat berbeda dari dataset.
- Sistem tidak mendeteksi nominal otomatis.
- Sistem tidak memvalidasi keaslian uang.
