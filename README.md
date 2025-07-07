# Klasifikasi Suara Hewan Menggunakan CNN (Convolutional Neural Network)

Ini adalah proyek *machine learning* yang mengimplementasikan *Convolutional Neural Network* (CNN) untuk mengklasifikasikan suara-suara hewan (moo, meow, woof, mbee, tweet). Model dilatih menggunakan *dataset* audio yang direkam secara langsung, kemudian dievaluasi, dan dapat digunakan untuk prediksi suara baru.

---

## Fitur Utama

* **Pengumpulan Data Audio**: Dataset dibuat dari rekaman suara pribadi yang menirukan suara hewan.
* **Pra-pemrosesan Audio**: Menerapkan *Voice Activity Detection* (VAD) untuk menghilangkan segmen hening, *padding/cropping* untuk konsistensi durasi, normalisasi amplitudo, dan ekstraksi *Mel-frequency Cepstral Coefficients* (MFCC).
* **Visualisasi Spektral**: Menampilkan bentuk gelombang (original dan *pre-processed*) serta MFCC untuk memahami fitur audio.
* **Analisis Formant**: Membandingkan formants (F1, F2) yang terdeteksi dengan rentang teoritis untuk memvalidasi kualitas data vokal.
* **Arsitektur CNN**: Menggunakan model CNN sederhana yang dibangun dengan PyTorch untuk tugas klasifikasi.
* **Pelatihan & Evaluasi Model**: Melatih model dan mengevaluasi kinerjanya menggunakan akurasi dan *classification report*.
* **Prediksi Suara Baru**: Model dapat memprediksi label untuk *file* audio baru dengan menampilkan tingkat kepercayaan untuk setiap kelas.

## Struktur Proyek
├── Speech Classification with CNN.ipynb  # Notebook utama proyek
├── animal_cnn.pth                        # File model yang sudah dilatih (akan diunggah)
├── data/                                 # Folder untuk data training
│   ├── moo/
│   ├── meow/
│   ├── woof/
│   ├── mbee/
│   └── tweet/
└── test/                                 # Folder untuk data uji coba
    ├── cat_meow.wav
    ├── cow_moo.wav
    ├── dog_woof.wav
    ├── goat_mbee.wav
    └── bird_twit.wav


### Pra-pemrosesan Data

Tahap pra-pemrosesan meliputi:
-   **Voice Activity Detection (VAD)**: Menggunakan `librosa.effects.split` untuk menghapus bagian audio yang hening.
-   **Normalisasi Durasi**: Audio di-*padding* atau di-*crop* agar memiliki durasi yang konsisten (2 detik).
-   **Normalisasi Amplitudo**: Amplitudo sinyal audio dinormalisasi ke rentang [-1, 1].
-   **Ekstraksi Fitur MFCC**: Melakukan ekstraksi 40 koefisien MFCC, yang kemudian distandarisasi (zero mean, unit variance).

Visualisasi bentuk gelombang (original dan *pre-processed*) serta *spectrogram* MFCC disertakan dalam *notebook* untuk menunjukkan efektivitas pra-pemrosesan dalam membersihkan dan menstandarisasi data. Analisis formant (F1 dan F2) juga dilakukan untuk memvalidasi kualitas data vokal secara teoritis.

## Arsitektur Model (CNN)

Model menggunakan arsitektur CNN sederhana yang dibangun dengan PyTorch:
-   Dua lapisan konvolusi 2D (`nn.Conv2d`), masing-masing diikuti oleh fungsi aktivasi ReLU dan lapisan *MaxPooling* (`nn.MaxPool2d`).
-   Lapisan *fully connected* (`nn.Linear`) di bagian akhir untuk klasifikasi ke dalam 5 kelas vokal.

## Hasil

Model dilatih selama 100 *epoch* menggunakan *optimizer* Adam dan *CrossEntropyLoss*.

-   **Akurasi Training**: 100.00%
-   **Akurasi Test**: 96.00%

*Classification Report* menunjukkan kinerja model yang sangat baik untuk sebagian besar kelas. Untuk vokal 'woof', *recall* sedikit lebih rendah (0.80), dan untuk vokal 'tweet', *precision* sedikit lebih rendah (0.83). Secara keseluruhan, *macro avg* dan *weighted avg f1-score* mencapai 0.96. Ini menunjukkan bahwa model cukup efektif dalam tugas klasifikasi ini pada dataset yang digunakan.

Hasil uji coba prediksi pada *file* terpisah menunjukkan bahwa model mampu mengklasifikasikan sebagian besar suara vokal dengan tingkat kepercayaan yang sangat tinggi.

## Cara Menggunakan

1.  **Kloning Repositori**:
    ```bash
    git clone [https://github.com/nabilahpw/speech_classification.git](https://github.com/nabilahpw/speech_classification.git)
    cd speech_classification
    ```

2.  **Siapkan Dataset**:
    * Pastikan Anda memiliki *dataset* audio yang terstruktur seperti di bagian "[Struktur Proyek](#struktur-proyek)".
    * *Upload* folder `data` dan `test` ke Google Drive Anda di *path* `/My Drive/speech_classification/`.

3.  **Buka Notebook di Google Colab**:
    * Buka `Speech Classification with CNN.ipynb` di Google Colab.

4.  **Jalankan Sel-sel Notebook**:
    * Jalankan sel pertama untuk *mounting* Google Drive Anda. Ikuti instruksi otorisasi.
    * Pastikan *path* `DATA_DIR` di sel "Parameters" mengarah ke lokasi data Anda di Google Drive: `/content/drive/My Drive/speech_classification/data`.
    * Jalankan semua sel secara berurutan. *Notebook* ini dirancang untuk berjalan dari awal hingga akhir.

5.  **Uji Coba Prediksi**:
    * Pada bagian "UJI COBA KLASIFIKASI SUARA VOKAL DENGAN MODEL YANG TELAH DIBUAT", kode akan secara otomatis menguji *file* `.wav` yang ada di folder `/content/drive/My Drive/speech_classification/test`.

## Keterbatasan

* **Ukuran Dataset**: Dataset yang digunakan relatif kecil (15 sampel per kelas). Hal ini dapat menyebabkan *overfitting* dan membuat model kurang generalisir pada data yang sangat bervariasi.
* **Variasi Data**: Dataset ini direkam secara pribadi dengan kondisi yang mungkin tidak bervariasi (misalnya, satu pembicara, satu lingkungan).
* **Analisis Formant**: Analisis formant digunakan untuk validasi kualitas data, tetapi untuk analisis lebih mendalam, alat profesional dan dataset yang lebih luas akan dibutuhkan.
