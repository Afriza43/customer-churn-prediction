# Laporan Proyek Machine Learning - Afriza Meidio Andhana

## Domain Proyek

Kehilangan pelanggan merupakan masalah besar bagi industri perbankan. Inovasi teknologi yang berkembang saat ini memungkinkan pelanggan untuk dengan mudah membuka rekening baru atau memindahkan aset mereka ke bank lain tanpa perlu datang secara fisik. Situasi ini menciptakan tantangan besar bagi bank dalam menjaga loyalitas pelanggan. Menurut Damanik et al. (2023), perusahaan perlu memahami dengan baik pelanggan mana yang cenderung berhenti agar dapat menyusun strategi retensi yang tepat sebelum kehilangan pelanggan tersebut. Dalam mempertahankan dan menarik customer baru tidak dapat dilakukan secara acak, ada beberapa faktor yang harus diperhatikan dalam menarik customer secara efektif. Oleh karena itu perlu diketahui faktor penyebab churn dalam upaya peningkatan retensi pelanggan di perusahaan (Husein et al., 2021)

Customer churn merupakan salah satu masalah besar yang dihadapi oleh industri perbankan. Menurut riset oleh Harvard Business Review, menarik pelanggan baru memerlukan biaya yang jauh lebih tinggi dibandingkan mempertahankan pelanggan yang sudah ada. Oleh karena itu, bank sangat tertarik untuk mengetahui faktor-faktor apa yang menyebabkan pelanggan memutuskan untuk meninggalkan layanan mereka. Dengan memahami alasan di balik churn, bank dapat mengembangkan program loyalitas dan kampanye retensi untuk mempertahankan pelanggan mereka. Dengan demikian, prediksi churn dengan akurasi tinggi bisa sangat membantu dalam menyusun strategi retensi yang efektif agar perusahaan dapat menjaga retensi pelanggan. Untuk memprediksi churn perlu implementasi ilmu machine learning agar mencapai hasil prediksi yang akurat.

Referensi:

- [Klasifikasi customer churn pada telekomunikasi industri untuk retensi pelanggan menggunakan algoritma C4.5](http://www.djournals.com/klik/article/view/829)
- [Pendekatan data science untuk menemukan churn pelanggan pada sektor perbankan dengan machine learning](https://jurnal.itscience.org/index.php/dsi/article/view/1169)
- [The value of keeping the right customers. Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)

## Business Understanding

Sebuah bank perlu sebuah sistem yang dapat melakukan prediksi customer churn, serta faktor-faktor apa saja yang dapat membuat pelanggan meninggalkan bank tersebut. Dengan demikian, bank dapat memprediksi pelanggan yang mungkin akan meninggalkan bank dan mengambil keputusan untuk menjaga retensi pelanggan.

### Problem Statements

Dari latar belakang tersebut, maka dibuat pernyataan masalah sebagai berikut :

- Bagaimana cara memprediksi apakah seorang pelanggan akan meninggalkan bank atau tidak berdasarkan data historis pelanggan?
- Dari beberapa faktor yang ada, faktor apa saja yang paling berpengaruh terhadap keputusan pelanggan untuk pergi dari Bank?

### Goals

Tujuan dari pernyataan masalah:

- Membangun model machine learning yang dapat memprediksi churn pelanggan secara akurat.
- Mengidentifikasi faktor-faktor utama yang menyebabkan pelanggan pergi dari Bank.

  ### Solution statements

  - Menggunakan beberapa algoritma machine learning untuk memprediksi customer churn secara akurat. Algoritma klasifikasi yang digunakan yaitu: Random Forest, Decision Tree, XGBoost, dan CatBoost
  - Menganalisis data lebih dalam untuk mengetahui faktor apa saja yang mempengaruhi terjadinya customer churn

## Data Understanding

Data yang digunakan dalam proyek ini adalah data history sebuah bank di Eropa yang berisi mengenai aktivitas dan status pelanggan mereka. Dataset ini memiliki 10.000 baris data yang menunjukkan status customer apakah sudah meninggalkan bank atau belum, dengan berbagai karakteristik. Karakteristik disini adalah RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited, Complain, Satisfaction Score, ard Type, Points Earned.

Sumber Dataset : [Bank Customer Churn - Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data)

### Variabel-variabel pada Bank Customer Churn adalah sebagai berikut:

Dataset terdiri dari 14 kolom atau fitur yang memberikan informasi mengenai pelanggan dan perilaku mereka:

- RowNumber : Urutan baris.
- CustomerId : ID pelanggan.
- Surname : Nama belakang pelanggan.
- CreditScore : Skor kredit pelanggan dalam melunasi kredit, semakin tinggi semakin baik.
- Geography : Lokasi geografis pelanggan.
- Gender : Jenis kelamin pelanggan.
- Age : Usia pelanggan.
- Tenure : Lama pelanggan menjadi nasabah bank.
- Balance : Saldo rekening pelanggan.
- NumOfProducts : Jumlah produk yang dibeli dengan bank tersebut oleh pelanggan.
- HasCrCard : Apakah pelanggan memiliki kartu kredit.
- IsActiveMember : Apakah pelanggan aktif menggunakan layanan bank.
- EstimatedSalary : Gaji yang diperkirakan dari pelanggan.
- Exited : Apakah pelanggan meninggalkan bank (label target).
- Complain : pelanggan mempunyai keluhan atau tidak.
- Satisfaction Score : Skor yang diberikan oleh pelanggan untuk penyelesaian keluhan mereka.
- Card Type : jenis kartu yang dipegang oleh pelanggan.
- Point Earned : poin yang diperoleh pelanggan karena menggunakan kartu kredit.

### Exploratory Data Analysis

Terdapat 4 kolom dengan tipe object, yaitu: Surname, Geography, Gender, dan Card Type. Kolom ini merupakan categorical features (fitur non-numerik).
Terdapat 14 kolom numerik dengan tipe data int64 dan float64 yaitu: RowNumber, CustomerID, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited, Complain, Satisfaction Score, Point Earned.
Kolom "Exited" adalah target dari prediksi ini.

Terdapat beberapa fitur yang tidak digunakan untuk mengembangkan model. Fitur yang dihapus adalah fitur 'RowNumber', 'CustomerID', 'Surname'.

Setelah itu, dilakukan pengecekan apakah terdapat nilai kosong, data duplikat dan outlier pada dataset tersebut. Dari hasil analisis, ternyata dataset sudah cukup bersih karena tidak ditemukan adanya nilai kosong dan juga tidak ada data yang terduplikat.

Terdapat beberapa outlier pada fitur 'Age' dan 'CreditScore'. Akan tetapi, kali ini outliers tidak akan di hapus karena data tersebut masih direntang yang masuk akal dan masih mengandung informasi penting untuk prediksi Customer Churn

#### Univariate Analysis

**Fitur Kategorikal**

1. Gender

   ![Bar Chart Gender](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-uni-gender.png)

Berdasarkan grafik di atas, customer laki - laki lebih banyak, sekitar 54,6% dari seluruh data, daripada perempuan sebesar 45,4%.

2. Card Type

   ![Bar Chart Card Type](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-uni-cardtype.png)

Terdapat 4 kategori tipe kartu yang dimiliki oleh customer, yaitu Diamond, Gold, Silver, Platinum. Masing - masing kategori tersebut memiliki jumlah yang sama yaitu 25% dari jumlah dataset, sekitar 2500 customer untuk masing - masing kategori

3. Geography

   ![Bar Chart Geography](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-uni-geo.png)

Pada dataset, customer hanya berasal dari 3 negara, diantaranya France, Spain, dan German. Dari grafik di atas, dapat dilihat bahwa Bank tersebut banyak yang berasal dari negara France, sekitar 50% dari sampel. Sedangkan sisanya, yaitu berasal dari Germany dan Spain sebanyak 25% dari data untuk masing - masing negara

**Fitur Numerik**

Hubungan Antar Fitur Numerik

![Histogram](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/histogram_fitur_numerik.png)

Dari grafik di atas, maka dapat disimpulkan sebagai berikut :

- Skor pelanggan dalam melunasi kredit berkisar di antara 600 - 700, yang berarti rata - rata nasabah cukup cepat melunasi kredit
- Customer bank tersebut kebanyakan berumur 35 - 40 tahun
- Cukup banyak pelanggan yang tidak memiliki saldo di rekening bank tersebut
- Hampir 50% pelanggan pada sampel sudah tidak aktif menggunakan layanan bank tersebut
- Gaji pelanggan pada bank tersebut cukup variatif dari 0 - 200.000
- 20% pelanggan dari sampel mengeluh dengan bank tersebut.

#### Multivariate Analysis

**Fitur Kategorikal**

1. Hubungan Fitur Gender dengan Target (Exited)

   ![Bar Chart Gender dengan Target](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-multi-gender.png)

Dari grafik, dapat dilihat bahwa banyak pelanggan yang keluar berasal dari negara German dan France

2. Hubungan Fitur Geography dengan Target (Exited)

   ![Bar Chart Geography dengan Target](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-multi-geo.png)

Dari semua pelanggan perempuan, 25% perempuan memutuskan untuk meninggalkan bank.Dari semua pelanggan laki - laki, 16% meninggalkan bank

3. Hubungan Fitur Card Type dengan Target (Exited)

   ![Bar Chart Card Type dengan Target](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-multi-cardtype.png)

Sekitar 20% pelanggan dari tiap kategori kartu memutuskan untuk pergi meninggalkan bank

**Fitur Numerik**

1. Matrik Heatmap
   ![Heatmap Fitur Numerik](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/corr_matrix.png)

Jika diperhatikan dengan baik, fitur 'Complain' memiliki skor korelasi dengan target 'Exited' sebesar 1. Artinya, hampir setiap pelanggan yang meninggalkan bank tersebut pasti memiliki keluhan dengan bank tersebut.

Selain itu, adapun fitur 'Age' dan 'Balance' yang memiliki korelasi dengan fitur 'Exited' meskipun korelasi nya rendah. Dapat dikatakan bahwa umur pelanggan dan jumlah saldo rekening pelanggan dapat sedikit mempengaruhi apakah pelanggan meninggalkan bank atau tidak.

Tetapi terdapat fitur 'IsActiveMember' yang berkorelasi negatif dengan fitur 'Exited'. Berarti kedua fitur tersebut berbanding terbalik nilainya meskipun korelasi nya lemah.

Selain fitur tersebut, semua fitur lainnya memiliki korelasi yang kecil. Sehingga tidak diperlukan untuk melakukan prediksi dan dapat dihapus

2. Hubungan Status Aktif Customer dengan Target (Exited)
   ![Bar Chart IsActiveMember dengan Target](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-multi-member_status.png)

Member yang tidak aktif lebih banyak meninggalkan bank daripada yang masih aktif

3. Hubungan Umur dengan Target (Exited)
   ![Bar Chart Umur dengan Target](https://github.com/Afriza43/customer-churn-prediction/blob/main/images/barchart-multi-age.png)

Pelanggan dalam kelompok usia 50-60 tahun memiliki tingkat churn tertinggi, yaitu 56.21%. Ini berarti lebih dari separuh pelanggan dalam kelompok usia ini memutuskan untuk meninggalkan bank.

## Data Preparation

Dalam proyek ini, proses data preparation dilakukan dengan beberapa tahapan untuk memastikan bahwa data siap digunakan dalam pemodelan machine learning. Tahapan ini dilakukan dengan berurutan untuk memastikan hasil yang optimal dari algoritma yang akan digunakan. Berikut adalah teknik-teknik data preparation yang diterapkan beserta alasan mengapa tahapan tersebut diperlukan:

1. Encoding Fitur Kategori
   Proses: Pada dataset yang digunakan, beberapa fitur bersifat kategorikal, seperti jenis keanggotaan dan keluhan pelanggan (IsActiveMember, Complain). Fitur-fitur ini perlu diubah menjadi nilai numerik agar dapat diproses oleh model machine learning. Teknik One-Hot Encoding digunakan untuk mengonversi nilai kategorikal menjadi fitur numerik biner.

   Alasan: Algoritma machine learning sebagian besar bekerja dengan data numerik, sehingga fitur kategorikal harus diubah agar model dapat memahaminya. One-Hot Encoding digunakan karena teknik ini menciptakan representasi biner yang sesuai untuk variabel kategorikal tanpa menciptakan hubungan ordinal antar nilai.

2. Handling Imbalance Data dengan Teknik Oversampling SMOTE
   Proses: Setelah pengkodean, dataset yang digunakan ternyata tidak seimbang. Terdapat lebih banyak data pelanggan yang tidak churn dibandingkan dengan pelanggan yang churn. Untuk mengatasi masalah ini, teknik SMOTE (Synthetic Minority Over-sampling Technique) diterapkan untuk menyeimbangkan data.

   Alasan: Ketidakseimbangan kelas pada data dapat menyebabkan model bias terhadap kelas mayoritas, yang mengurangi kemampuan model dalam memprediksi kelas minoritas dengan akurat. SMOTE meningkatkan jumlah sampel pada kelas minoritas dengan mensintesis data baru, yang membantu model untuk belajar secara lebih adil antara kelas mayoritas dan minoritas.

3. Splitting Data (80:20)
   Proses: Setelah data seimbang, langkah selanjutnya adalah membagi dataset menjadi dua bagian: data latih (train) dan data uji (test). Pembagian ini dilakukan dengan rasio 80:20, di mana 80% data digunakan untuk melatih model dan 20% data digunakan untuk menguji kinerja model.

   Alasan: Proses pemisahan data ini penting untuk menguji generalisasi model. Dengan melakukan pelatihan pada 80% data dan pengujian pada 20% sisanya, kita dapat mengevaluasi performa model di data yang belum pernah dilihat sebelumnya, memastikan bahwa model tidak overfitting terhadap data latih.

4. Standarisasi
   Proses: Standarisasi diterapkan pada fitur numerik seperti Age dan Balance. Teknik ini mengubah nilai setiap fitur sehingga memiliki rata-rata 0 dan varians 1, menggunakan StandardScaler. Fitur biner seperti IsActiveMember dan Complain tidak memerlukan standarisasi karena fitur ini hanya memiliki dua kategori, yaitu 0 dan 1.

   Alasan: Standarisasi sangat penting terutama ketika menggunakan algoritma yang sensitif terhadap skala data, seperti regresi logistik atau metode berbasis gradient descent (misalnya, XGBoost). Fitur dengan skala yang berbeda dapat menyebabkan model memberi bobot yang tidak proporsional pada fitur tertentu, sehingga standarisasi memastikan bahwa semua fitur memiliki skala yang sama.

## Modeling

Dalam proyek ini, empat algoritma machine learning diterapkan: Random Forest, Decision Tree, XGBoost, dan CatBoost. Setiap model dijalankan menggunakan parameter default, tanpa proses hyperparameter tuning. Berikut adalah tahapan pemodelan yang dilakukan, beserta kelebihan dan kekurangan dari masing-masing algoritma serta alasan pemilihan model terbaik.

1. Random Forest
   Tahapan: Random Forest adalah algoritma ensemble berbasis pohon keputusan. Algoritma ini membangun banyak pohon keputusan selama proses pelatihan dan memprediksi dengan cara mengambil rata-rata dari semua pohon (bagging). Pada proyek ini, Random Forest dijalankan dengan parameter default sebagai berikut:

- n_estimators=100 (jumlah pohon yang dibuat adalah 100)
- criterion='gini' (impuritas Gini digunakan untuk membagi node)
- max_depth=None (tidak ada batasan kedalaman pohon)
- random_state=None (tidak ada seed tertentu)

Kelebihan:

- Dapat menangani data dengan fitur yang besar tanpa mudah overfitting.
- Memiliki kemampuan untuk mengatasi masalah multikolinearitas pada data.
- Dapat memberikan estimasi fitur yang penting dalam pengambilan keputusan.

Kekurangan:

- Waktu pelatihan dan komputasi relatif lebih lama dibandingkan algoritma sederhana seperti Decision Tree.
- Model lebih sulit untuk diinterpretasikan secara langsung karena terdiri dari banyak pohon.

2. Decision Tree
   Tahapan: Decision Tree adalah algoritma pembelajaran yang bekerja dengan membagi data secara rekursif ke dalam subset berdasarkan nilai fitur yang berbeda. Decision Tree dalam proyek ini juga dijalankan dengan parameter default:

- criterion='gini' (menggunakan Gini index sebagai pengukuran kemurnian node)
- splitter='best' (memilih pemisahan terbaik pada setiap node)
- max_depth=None (tidak ada batasan pada kedalaman pohon)

Kelebihan:

- Mudah dipahami dan diinterpretasikan.
- Cepat dalam melakukan pelatihan dan prediksi karena bersifat deterministik.

Kekurangan:

- Mudah overfitting pada data latih, terutama ketika pohon tumbuh sangat dalam tanpa batasan.
- Tidak memiliki mekanisme bawaan untuk menangani imbalance data.

3. XGBoost (Extreme Gradient Boosting)
   Tahapan: XGBoost adalah algoritma boosting yang kuat dan efisien yang menggabungkan banyak model pohon keputusan secara berurutan untuk meningkatkan akurasi prediksi. Dalam proyek ini, XGBoost dijalankan dengan parameter default:

- n_estimators=100 (jumlah pohon adalah 100)
- learning_rate=0.1 (kecepatan pembelajaran untuk setiap iterasi boosting)
- max_depth=6 (kedalaman maksimal setiap pohon adalah 6 level)
- random_state=None (tidak ada seed tertentu)

Kelebihan:

- Memiliki kinerja yang sangat baik pada dataset besar dan sering kali unggul dalam kompetisi machine learning.
- Memiliki mekanisme bawaan untuk mengatasi imbalance data dan overfitting.

Kekurangan:

- Sulit diinterpretasikan karena kompleksitas model yang tinggi.
- Memerlukan lebih banyak waktu komputasi dan sumber daya dibandingkan dengan algoritma lain.

4. CatBoost
   Tahapan: CatBoost adalah algoritma boosting yang dirancang khusus untuk menangani data kategorikal dengan baik tanpa perlu banyak praproses. CatBoost dijalankan dengan parameter default:

- iterations=1000 (jumlah maksimum iterasi adalah 1000)
- learning_rate=None (nilai kecepatan pembelajaran default ditentukan secara otomatis)
- depth=6 (kedalaman pohon maksimum adalah 6)
- verbose=1000 (menampilkan hasil iterasi setiap 1000 kali)

Kelebihan:

- Sangat efisien untuk menangani data kategorikal secara langsung.
- Menghasilkan performa tinggi dengan sedikit praproses data.

Kekurangan:

- Sama seperti XGBoost, CatBoost lebih sulit untuk diinterpretasikan.
- Diperlukan waktu yang lebih lama untuk melatih model dengan jumlah data besar.

### Model Terbaik: Random Forest

Berdasarkan hasil evaluasi dan metrik akurasi pada data uji, algoritma Random Forest dipilih sebagai model terbaik. Berikut adalah beberapa alasan mengapa Random Forest menjadi pilihan:

- Akurasi yang bagus: Random Forest memberikan hasil yang sangat baik dalam prediksi customer churn, dengan akurasi yang unggul dibandingkan algoritma lain.
- Kestabilan Model: Karena Random Forest adalah algoritma ensemble, ia lebih stabil dan cenderung tidak overfitting dibandingkan dengan Decision Tree.
- Kemampuan Fitur Importansi: Random Forest juga dapat memberikan wawasan tentang fitur mana yang paling berpengaruh terhadap prediksi churn.

Setiap algoritma dijalankan menggunakan parameter default tanpa hyperparameter tuning. Berikut adalah nilai parameter default untuk setiap algoritma yang digunakan:

- Random Forest: n_estimators=100, criterion='gini', max_depth=None
- Decision Tree: criterion='gini', splitter='best', max_depth=None
- XGBoost: n_estimators=100, learning_rate=0.1, max_depth=6
- CatBoost: iterations=1000, depth=6, verbose=1000

Meskipun tidak dilakukan tuning, hasil akurasi menunjukkan bahwa Random Forest memiliki performa yang cukup baik untuk memecahkan masalah ini.

## Evaluation

Pada proyek klasifikasi churn ini, metrik evaluasi yang digunakan adalah akurasi, precision, recall, dan F1 score. Metrik ini dipilih karena penting untuk mengevaluasi performa model klasifikasi dalam konteks churn, di mana keseimbangan antara prediksi positif dan negatif sangat penting.

![Confussion Matrix](https://miro.medium.com/v2/resize:fit:640/format:webp/0*QEC-f69bzuMd3cTD.png)

### Penjelasan Metrik yang Digunakan

#### Akurasi

**Akurasi**: Akurasi mengukur persentase prediksi yang benar dari keseluruhan prediksi. Formula untuk menghitung akurasi adalah:

![Rumus Akurasi](https://miro.medium.com/v2/resize:fit:828/format:webp/1*XjVhud9BW7vq5J_fUprnLg.png)

Di mana:

- **TP**: True Positives (Prediksi benar untuk kelas positif)
- **TN**: True Negatives (Prediksi benar untuk kelas negatif)
- **FP**: False Positives (Prediksi salah untuk kelas positif)
- **FN**: False Negatives (Prediksi salah untuk kelas negatif)

#### Precision

**Precision**: Precision mengukur seberapa baik model dalam memprediksi positif yang benar dibandingkan dengan semua prediksi positif. Formula precision adalah:

![Rumus Precision](https://miro.medium.com/v2/resize:fit:786/format:webp/1*DoGL8YNxBOwkX_gd9P_CEA.png)

Precision tinggi berarti model memberikan prediksi positif yang relevan dengan hasil aktual.

#### Recall

**Recall**: Recall mengukur kemampuan model untuk menemukan semua contoh positif dari keseluruhan positif yang ada. Formula recall adalah:

![Rumus Recall](https://miro.medium.com/max/538/1*OV0hfgCStTI8hy6lAY1SdA.jpeg)

Recall penting dalam kasus churn karena kita ingin mengidentifikasi semua pelanggan yang berpotensi churn.

#### F1 Score

**F1 Score**: F1 score adalah rata-rata harmonik dari precision dan recall. Metrik ini berguna ketika ada ketidakseimbangan antara kelas dan kita ingin menyeimbangkan precision dan recall. Formula F1 score adalah:

![Rumus F1 Score](https://ilmudatapy.com/wp-content/uploads/2021/01/confusion-matrix-5.png)

F1 score membantu kita melihat kinerja model secara menyeluruh, khususnya pada masalah klasifikasi yang memiliki distribusi kelas yang tidak seimbang.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Berikut adalah hasil evaluasi dari beberapa model yang digunakan (RandomForest, Decision Tree, XGBoost, dan CatBoost):

| Model             | Akurasi | Precision | Recall | F1 Score |
| ----------------- | ------- | --------- | ------ | -------- |
| **RandomForest**  | 0.996   | 0.993     | 0.999  | 0.996    |
| **Decision Tree** | 0.861   | 0.776     | 0.999  | 0.874    |
| **XGBoost**       | 0.949   | 0.905     | 0.999  | 0.950    |
| **CatBoost**      | 0.933   | 0.878     | 0.999  | 0.935    |

Dari tabel di atas, kita dapat melihat bahwa **Random Forest** adalah model dengan performa terbaik, dengan akurasi 99,6% dan F1 score 0,996. Hal ini menunjukkan bahwa model ini sangat akurat dalam memprediksi pelanggan yang berpotensi churn dan mempertahankan keseimbangan antara precision dan recall.

### Confusion Matrix

Berikut adalah nilai confussion matrix nya :

- True Negatives (0,0): 1640
- False Positives (0,1): 11
- False Negatives (1,0): 1
- True Positives (1,1): 1533

Dengan sangat sedikit kesalahan prediksi (FP dan FN yang rendah), Confusion Matrix ini menunjukkan bahwa model **RandomForest** mampu mengidentifikasi dengan baik pelanggan yang churn maupun yang tidak churn.

## Kesimpulan

Secara keseluruhan, model RandomForest dipilih sebagai model terbaik karena performanya yang unggul pada semua metrik evaluasi yang digunakan. Dengan demikian, model prediksi dapat digunakan oleh bank untuk memprediksi customer churn dengan akurat dan benar.

Faktor - faktor yang mempengaruhi pergi atau tidaknya seorang pelanggan adalah umur, saldo, status member (aktif/tidak aktif), dan yang paling mempengaruhi sekali adalah keluhan pelanggan. Jika pelanggan memiliki keluhan, maka hampir dipastikan pelanggan tersebut akan meninggalkan bank
