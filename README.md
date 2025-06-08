## 🧠 1. Business Understanding

### Problem Statements

* Bagaimana memprediksi risiko mahasiswa mengalami drop out dari perguruan tinggi berdasarkan data akademik dan non-akademik?
* Fitur-fitur apa yang paling signifikan dalam menentukan kemungkinan mahasiswa akan drop out?
* Bagaimana model prediksi dapat digunakan untuk memberikan peringatan dini agar institusi pendidikan dapat mengambil tindakan preventif?

### Goals

* Mengembangkan sistem prediksi drop out mahasiswa berbasis data historis.
* Menerapkan algoritma machine learning untuk mengklasifikasikan mahasiswa yang berisiko tinggi mengalami drop out.
* Mengevaluasi efektivitas model dalam mengidentifikasi mahasiswa berisiko menggunakan metrik akurasi, precision, recall, dan F1-score.

### Manfaat

* Membantu pihak kampus untuk mengenali mahasiswa yang berisiko drop out secara dini.
* Menyediakan dasar data bagi dosen pembimbing atau bagian akademik untuk memberikan intervensi tepat waktu.
* Meningkatkan tingkat kelulusan dan reputasi institusi pendidikan.
* Menjadi contoh penerapan machine learning dalam bidang pendidikan tinggi.

### Solusi yang Diterapkan

* **Pembuatan Data Sintetis**: Menggunakan `make_classification()` dari Scikit-learn untuk menciptakan dataset simulasi dengan proporsi kelas yang seimbang.
* **Pra-pemrosesan Data**:

  * Encoding fitur kategorikal dengan LabelEncoder.
  * Normalisasi fitur numerik menggunakan StandardScaler.
  * Pembagian data menjadi training dan testing.
  * Penanganan outliers menggunakan metode IQR (Interquartile Range) untuk menghapus atau mengoreksi data ekstrem yang dapat mengganggu pelatihan model.
  * Transformasi fitur dengan Principal Component Analysis (PCA) untuk mengurangi dimensi dan membantu model fokus pada fitur yang paling informatif.
* **Pemodelan**:

  * Menggunakan algoritma Random Forest untuk klasifikasi karena cocok untuk data tabular dan memiliki performa tinggi.
* **Evaluasi Model**:

  * Menggunakan berbagai metrik evaluasi seperti akurasi, precision, recall, dan F1-score untuk mengukur efektivitas prediksi.

---

## 📊 2. Data Understanding

### Sumber Data

Data dibuat secara sintetis menggunakan fungsi `make_classification()` dari `scikit-learn`, dan disesuaikan untuk mencerminkan kondisi nyata mahasiswa.

### Struktur Data

Dataset terdiri dari **500 mahasiswa**, masing-masing memiliki fitur:

* **Akademik**:

  * `GPA_Sem1` - `GPA_Sem8`: IPK tiap semester (1.5 – 4.0)

* **Non-Akademik**:

  * `Attendance_Rate`: Persentase kehadiran (%)
  * `Retaken_Courses`: Jumlah mata kuliah yang diulang
  * `LMS_Activity_Score`: Skor aktivitas e-learning
  * `Employment_Status`: Status kerja (Employed / Unemployed)
  * `Work_Hours`: Jam kerja per minggu
  * `Socioeconomic_Status`: Status ekonomi (Low / Middle / High)

* **Target**:

  * `Dropout`: 0 = Tidak dropout, 1 = Dropout

### Distribusi Data

* Data seimbang antara kelas `Dropout` = 0 dan 1 (50:50)

### Contoh Tampilan Dataset

| Student\_ID | GPA\_Sem1 | GPA\_Sem2 | GPA\_Sem3 | GPA\_Sem4 | GPA\_Sem5 | GPA\_Sem6 | GPA\_Sem7 | GPA\_Sem8 | Attendance\_Rate | Retaken\_Courses | LMS\_Activity\_Score | Employment\_Status | Work\_Hours | Socioeconomic\_Status | Dropout |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------------- | ---------------- | -------------------- | ------------------ | ----------- | --------------------- | ------- |
| MHS001      | 3.022455  | 2.806862  | 3.182229  | 2.345541  | 2.849578  | 3.008856  | 2.489718  | 2.997361  | 89.97            | 2                | 95.76                | Employed           | 10          | Middle                | 1       |
| MHS002      | 3.140474  | 3.519268  | 2.417743  | 2.479992  | 2.454279  | 2.091563  | 3.051875  | 2.196045  | 83.62            | 2                | 61.54                | Unemployed         | 13          | Middle                | 0       |
| MHS003      | 3.415879  | 3.080649  | 2.239178  | 2.840006  | 2.509998  | 2.414129  | 2.986644  | 2.239343  | 91.48            | 0                | 75.21                | Employed           | 20          | Low                   | 0       |
| MHS004      | 2.632963  | 2.531376  | 2.778220  | 2.600768  | 2.500836  | 2.820631  | 2.208334  | 3.119932  | 100.00           | 0                | 41.99                | Employed           | 13          | Middle                | 0       |
| MHS005      | 3.070948  | 3.090423  | 1.967964  | 2.705093  | 2.640558  | 2.207487  | 2.856562  | 2.435998  | 82.66            | 1                | 84.40                | Unemployed         | 28          | Middle                | 0       |

---

## 🧹 3. Data Preparation

### Langkah-langkah:

* **Encoding**:

  * `LabelEncoder` digunakan untuk fitur kategorikal seperti `Employment_Status` dan `Socioeconomic_Status`.

* **Outlier Handling**:

  * Digunakan metode IQR untuk mendeteksi dan mengatasi outlier pada fitur numerik, yang dapat memperburuk performa model jika dibiarkan.

* **Feature Engineering dengan PCA**:

  * Principal Component Analysis diterapkan untuk mereduksi dimensi data dan menjaga informasi utama, membantu meningkatkan efisiensi model dan mengurangi noise.

* **Splitting**:

  * Dataset dibagi menjadi **train-test** dengan rasio 80:20, memastikan model dapat dievaluasi pada data yang tidak terlihat sebelumnya.

* **Scaling dan Transformasi**:

  * Setelah data di-split, fitur numerik pada data training dan testing dinormalisasi dengan `StandardScaler`. Hal ini penting agar model tidak bias terhadap fitur dengan skala lebih besar.
  * PCA juga diterapkan pasca-scaling untuk memastikan transformasi dilakukan dengan konsisten pada ruang vektor yang terstandarisasi.

---

## 🤖 4. Modeling

### Model yang Digunakan

* **Random Forest Classifier**

### Penjelasan Algoritma

Random Forest merupakan algoritma ensemble learning yang membangun banyak pohon keputusan (decision tree) pada berbagai subset data, lalu menggabungkan hasil prediksinya melalui voting (klasifikasi) atau rata-rata (regresi).

Keunggulan utama Random Forest:

* Dapat menangani dataset dengan fitur numerik dan kategorikal tanpa perlu banyak pra-pemrosesan.
* Tahan terhadap overfitting karena menggabungkan hasil dari banyak pohon.
* Menyediakan metrik penting seperti feature importance untuk menginterpretasi model.

### Alasan Pemilihan

* **Kinerja Baik untuk Data Tabular**: Random Forest sangat efektif untuk data tabular seperti dataset mahasiswa.
* **Robust terhadap Outlier dan Missing Values**: Memberikan toleransi terhadap data yang tidak sempurna.
* **Interpretabilitas**: Fitur penting dapat diidentifikasi, sehingga bermanfaat bagi institusi dalam mengetahui faktor-faktor utama dropout.
* **Cepat dan Stabil**: Training time relatif singkat dan dapat di-paralelkan.

---

## 📈 5. Evaluation

### Metrik Evaluasi:

* **Akurasi**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC AUC Score**
* **Confusion Matrix**

### Hasil Evaluasi (berdasarkan kode notebook):

> *Silakan konfirmasi jika ingin saya jalankan sel evaluasi model agar hasil akurasi dan metrik lainnya bisa dicantumkan di sini.*

---

## 🚀 6. Deployment

### Status

* Model belum dideploy ke sistem produksi.
* Namun dapat disiapkan untuk:

  * Integrasi ke dashboard monitoring.
  * Pengujian lebih lanjut dengan data nyata.

---

### ✅ Catatan Tambahan:

Jika Anda ingin laporan ini dalam bentuk **dokumen Word / PDF**, atau ingin **diagram alur CRISP-DM** juga disertakan, saya bisa bantu buatkan.
