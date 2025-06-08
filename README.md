## 🤠 1. Business Understanding

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
![distribusi numerik ]()
![distribusi kategorikal ]()
> Distribusi setiap variabel cukup bagus setelah outliersnya ditangani
***
### Contoh Tampilan Dataset

| Student\_ID | GPA\_Sem1 | GPA\_Sem2 | GPA\_Sem3 | GPA\_Sem4 | GPA\_Sem5 | GPA\_Sem6 | GPA\_Sem7 | GPA\_Sem8 | Attendance\_Rate | Retaken\_Courses | LMS\_Activity\_Score | Employment\_Status | Work\_Hours | Socioeconomic\_Status | Dropout |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------------- | ---------------- | -------------------- | ------------------ | ----------- | --------------------- | ------- |
| MHS001      | 3.022455  | 2.806862  | 3.182229  | 2.345541  | 2.849578  | 3.008856  | 2.489718  | 2.997361  | 89.97            | 2                | 95.76                | Employed           | 10          | Middle                | 1       |
| MHS002      | 3.140474  | 3.519268  | 2.417743  | 2.479992  | 2.454279  | 2.091563  | 3.051875  | 2.196045  | 83.62            | 2                | 61.54                | Unemployed         | 13          | Middle                | 0       |
| MHS003      | 3.415879  | 3.080649  | 2.239178  | 2.840006  | 2.509998  | 2.414129  | 2.986644  | 2.239343  | 91.48            | 0                | 75.21                | Employed           | 20          | Low                   | 0       |
| MHS004      | 2.632963  | 2.531376  | 2.778220  | 2.600768  | 2.500836  | 2.820631  | 2.208334  | 3.119932  | 100.00           | 0                | 41.99                | Employed           | 13          | Middle                | 0       |
| MHS005      | 3.070948  | 3.090423  | 1.967964  | 2.705093  | 2.640558  | 2.207487  | 2.856562  | 2.435998  | 82.66            | 1                | 84.40                | Unemployed         | 28          | Middle                | 0       |


---

## 🪹 3. Data Preparation

### Langkah-langkah:

* **Encoding**:

  * `LabelEncoder` digunakan untuk fitur kategorikal seperti `Employment_Status` dan `Socioeconomic_Status` agar dapat digunakan dalam model ML.

* **Outlier Handling**:

  * Digunakan metode IQR untuk mendeteksi dan menghapus outlier pada fitur numerik seperti `Attendance_Rate`, `LMS_Activity_Score`, dan `Work_Hours`. Outlier yang berada di luar rentang Q1 - 1.5*IQR atau Q3 + 1.5*IQR dihapus untuk menjaga konsistensi distribusi data.

![heatmap ]()
> Distribusi setiap variabel cukup bagus setelah outliersnya ditangani

* **Feature Engineering dengan PCA**:

  * Principal Component Analysis (PCA) digunakan untuk mereduksi dimensi dari fitur numerik agar model dapat lebih fokus pada variasi utama dalam data. Dua komponen utama (principal components) digunakan sebagai fitur tambahan.

![hasil pca ]()
> Distribusi setiap variabel cukup bagus setelah outliersnya ditangani

* **Splitting**:

  * Dataset dibagi menjadi **training set (80%)** dan **testing set (20%)** untuk mengevaluasi generalisasi model terhadap data yang belum terlihat sebelumnya.

* **Scaling dan Transformasi**:

  * Setelah data di-split, fitur numerik di-training set dan test set dinormalisasi dengan `StandardScaler` untuk memastikan skala seragam.
  * PCA kemudian diterapkan pada hasil scaling untuk memastikan konsistensi transformasi dalam ruang fitur terstandarisasi.

### Statistik Deskriptif X\_test Setelah Transformasi

|           | Attendance\_Rate | Retaken\_Courses | LMS\_Activity\_Score | Work\_Hours | principal component 1 | principal component 2 |
| --------- | ---------------- | ---------------- | -------------------- | ----------- | --------------------- | --------------------- |
| **count** | 150.000000       | 150.000000       | 150.000000           | 150.000000  | 150.000000            | 150.000000            |
| **mean**  | 0.027599         | -0.064618        | -0.076259            | 0.037951    | 0.080196              | -0.052264             |
| **std**   | 1.080562         | 0.998055         | 1.011863             | 0.974867    | 0.909342              | 0.982594              |
| **min**   | -3.014058        | -1.306229        | -2.940263            | -1.606092   | -2.692334             | -2.452117             |
| **25%**   | -0.813761        | -1.078550        | -0.734873            | -0.707856   | -0.605996             | -0.716297             |
| **50%**   | 0.110367         | -0.395512        | -0.040711            | 0.135941    | 0.137851              | -0.089473             |
| **75%**   | 0.840090         | 0.515206         | 0.626226             | 0.761984    | 0.711575              | 0.550795              |
| **max**   | 1.726464         | 1.881282         | 1.585400             | 2.422359    | 2.397943              | 2.878029              |

---

## 🤖 4. Modeling

### Model yang Digunakan

* **Random Forest Classifier**

### Penjelasan Algoritma

Random Forest adalah algoritma berbasis ensemble learning yang membangun banyak pohon keputusan (decision tree) menggunakan subset acak dari data dan fitur. Setiap pohon memberikan satu prediksi, dan prediksi akhir ditentukan berdasarkan voting mayoritas (untuk klasifikasi).

Karakteristik utama Random Forest:

* Menggabungkan banyak pohon untuk meningkatkan akurasi dan stabilitas.
* Mengurangi risiko overfitting dibanding decision tree tunggal.
* Memberikan feature importance, memungkinkan kita memahami variabel apa yang paling berkontribusi terhadap prediksi.
* Mendukung data numerik dan kategorikal, serta toleran terhadap missing values.

### Alasan Pemilihan

* **Kinerja Tinggi di Data Tabular**: Random Forest dikenal memiliki performa sangat baik dalam menangani dataset tabular dengan banyak fitur.
* **Robust terhadap Outliers dan Noise**: Dengan agregasi dari banyak pohon, model lebih stabil dan tidak terlalu terpengaruh data ekstrem.
* **Kemampuan Generalisasi Baik**: Cocok untuk menghindari overfitting meskipun data relatif kecil.
* **Interpretabilitas**: Dapat digunakan untuk analisis feature importance untuk memahami pola dropout mahasiswa.
* **Mudah Diimplementasikan dan Cepat**: Library `scikit-learn` menyediakan implementasi Random Forest yang efisien dan mudah digunakan.

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


![Confusion Matrix ]()
> Distribusi setiap variabel cukup bagus setelah outliersnya ditangani


---

## 🚀 6. Deployment
