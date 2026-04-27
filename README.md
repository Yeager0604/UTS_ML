# 🍊 Klasifikasi Jeruk (Orange) vs Grapefruit
### Perbandingan Tiga Model: Decision Tree | Naive Bayes | Support Vector Machine

---

## 📌 Deskripsi Proyek

Proyek ini membandingkan tiga algoritma machine learning klasik untuk
mengklasifikasikan buah jeruk (orange) dan grapefruit berdasarkan
fitur fisik dan warna buah.

**Sumber Dataset:** [Kaggle – joshmcadams/oranges-vs-grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)

---

## 📋 Deskripsi Dataset

| Kolom      | Tipe    | Keterangan                               |
|------------|---------|------------------------------------------|
| `diameter` | float   | Diameter buah dalam satuan cm            |
| `weight`   | float   | Berat buah dalam satuan gram             |
| `red`      | float   | Nilai saluran warna merah (RGB: 0–255)   |
| `green`    | float   | Nilai saluran warna hijau (RGB: 0–255)   |
| `blue`     | float   | Nilai saluran warna biru (RGB: 0–255)    |
| `name`     | string  | Label kelas: `orange` atau `grapefruit`  |

Total sampel: **10.000** (5.000 orange, 5.000 grapefruit)

---

## 🔄 Tahapan Pembuatan Model

### Tahap 1 — Memuat Dataset
- Membaca `citrus.csv` menggunakan Pandas
- Menampilkan shape, distribusi kelas, dan statistik deskriptif

### Tahap 2 — Preprocessing
- **Cek missing values** → tidak ada nilai kosong
- **Label Encoding:** `grapefruit = 0`, `orange = 1`
- **Train-Test Split:** 80% training / 20% testing (stratified)
- **StandardScaler:** normalisasi fitur untuk Naive Bayes dan SVM

### Tahap 3 — Eksplorasi Data (EDA)
- Histogram distribusi setiap fitur berdasarkan kelas
- Heatmap korelasi antar fitur
- Output: `eda_distributions.png`, `eda_correlation.png`

### Tahap 4 — Training & Evaluasi Model

#### 🌳 Decision Tree
> Algoritma pohon keputusan yang mempartisi data berdasarkan nilai fitur
> secara rekursif hingga mencapai daun (leaf) berisi label kelas.

- Parameter: `max_depth=5`, `criterion='gini'`
- Tidak memerlukan normalisasi fitur
- Keunggulan: mudah diinterpretasi, cepat
- Kelemahan: rentan overfitting jika kedalaman tidak dibatasi

**Metrik:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, CV 5-fold

---

#### 📊 Naive Bayes (GaussianNB)
> Algoritma probabilistik berdasarkan Teorema Bayes dengan asumsi
> bahwa setiap fitur bersifat independen satu sama lain.

- Menggunakan distribusi Gaussian untuk setiap fitur
- Memerlukan normalisasi fitur (StandardScaler)
- Keunggulan: sangat cepat, cocok sebagai baseline
- Kelemahan: asumsi independensi fitur jarang terpenuhi di dunia nyata

**Metrik:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, CV 5-fold

---

#### ⚙️ Support Vector Machine (SVM)
> Algoritma yang mencari hyperplane optimal untuk memisahkan dua kelas
> dengan margin terbesar di ruang fitur berdimensi tinggi.

- Kernel: `RBF (Radial Basis Function)`, C=1.0, gamma='scale'
- Memerlukan normalisasi fitur (StandardScaler)
- Keunggulan: akurasi tinggi, efektif di ruang dimensi tinggi
- Kelemahan: lambat pada dataset besar, perlu tuning hyperparameter

**Metrik:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, CV 5-fold

---

### Tahap 5 — Perbandingan Model
Tabel ringkasan Accuracy, ROC-AUC, dan CV Accuracy ketiga model
untuk menentukan model terbaik.

### Tahap 6 — Visualisasi Hasil
File grafik yang dihasilkan:

| File                       | Isi                                    |
|----------------------------|----------------------------------------|
| `eda_distributions.png`    | Histogram distribusi fitur per kelas   |
| `eda_correlation.png`      | Heatmap korelasi antar fitur           |
| `confusion_matrices.png`   | Confusion matrix ketiga model          |
| `roc_curves.png`           | Kurva ROC ketiga model dalam satu plot |
| `model_comparison.png`     | Bar chart perbandingan metrik          |
| `decision_tree_viz.png`    | Visualisasi struktur pohon keputusan   |

### Tahap 7 — Prediksi Data Baru
Demonstrasi prediksi buah baru menggunakan ketiga model yang telah dilatih.
