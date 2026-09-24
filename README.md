# x - Aplikasi Real-Time Face Recognition (InsightFace)


```sh
python app.py --model buffalo_l --source http://192.168.1.50:8080/video
```

Aplikasi pengenalan wajah (*face recognition*) real-time berbasis **Python** dan **InsightFace**. Aplikasi ini dirancang agar dapat langsung digunakan menggunakan **webcam laptop** ataupun **kamera HP** (via Wi-Fi / IP Webcam stream), dengan performa tinggi dan akurat bahkan di CPU laptop biasa.

---

## 🌟 Fitur Utama

1. **Akurasi State-of-the-Art (InsightFace)**:
   - Menggunakan deteksi wajah RetinaFace dan ekstraksi fitur ArcFace.
   - Sangat tahan terhadap sudut miring wajah, ekspresi, dan variasi pencahayaan.
2. **Ringan & Cepat di CPU Laptop**:
   - Menggunakan model teroptimasi `buffalo_sc` (hanya ~15 MB), memberikan FPS tinggi tanpa memerlukan GPU khusus.
3. **Estimasi Umur & Jenis Kelamin (Age & Gender)**:
   - Dengan model `buffalo_l`, sistem otomatis memprediksi perkiraan usia (umur) dan gender setiap wajah di kamera secara live (misal: `[33th, L]`).
4. **Fleksibel untuk Berbagai Sumber Kamera**:
   - Langsung mendeteksi webcam laptop bawaan (indeks `0`).
   - Mendukung streaming kamera HP secara wireless lewat aplikasi seperti *IP Webcam*, *DroidCam*, atau *RTSP/HTTP stream*.
5. **Pendaftaran Wajah Sangat Mudah (3 Cara)**:
   - **Live In-App**: Tekan tombol **`S`** saat kamera aktif untuk langsung mendaftarkan wajah.
   - **Skrip Registrasi Interaktif**: Jalankan `python register.py` dengan panduan visual.
   - **Folder Manual**: Cukup letakkan file foto di dalam folder `targets/` (misal: `targets/Budi.jpg`).
6. **Tampilan Visual Modern**:
   - Desain bounding box dengan aksen sudut (*corner brackets*).
   - Tag label nama orang, umur, gender, dan persentase skor kemiripan (*similarity score*).
   - HUD atas menampilkan FPS real-time, status model, dan jumlah wajah terdaftar.
   - Hot-reload database wajah secara instan tanpa perlu mematikan aplikasi (tekan tombol **`R`**).

---

## 📁 Struktur Direktori

```text
x/
├── app.py                 # Aplikasi utama Face Recognition real-time
├── register.py            # Skrip pendaftaran wajah (kamera / impor foto)
├── face_system.py         # Modul inti (InsightFace, Cosine Similarity, Overlay UI)
├── requirements.txt       # Daftar paket dependensi Python
├── targets/               # Folder database foto wajah yang didaftarkan
│   └── PETUNJUK_DATABASE.txt
└── README.md              # Dokumentasi lengkap proyek
```

---

## 🚀 Instalasi & Persiapan

### 1. Prasyarat
- Python 3.8 hingga 3.12 (Aplikasi ini sudah dites dan berjalan di Python 3.12 pada Windows 11).

### 2. Instalasi Dependensi
Buka Terminal atau PowerShell di folder ini, lalu jalankan:

```powershell
pip install -r requirements.txt
```

> **Catatan**: Saat pertama kali dijalankan, model InsightFace (`buffalo_sc`) akan otomatis diunduh ke komputer Anda (~14.6 MB) dan disimpan untuk penggunaan offline selanjutnya.

---

## 📸 Cara Menjalankan

### A. Menggunakan Webcam Laptop Bawaan
Cukup jalankan perintah:

```powershell
python app.py
```
Aplikasi akan otomatis mendeteksi webcam bawaan laptop Anda.

---

### B. Menggunakan Kamera HP (Android / iOS)
Anda bisa menjadikan kamera HP beresolusi tinggi sebagai sumber kamera komputer via Wi-Fi:

1. **Install aplikasi streaming di HP**:
   - **Android**: Install aplikasi gratis **IP Webcam** (oleh Pavel Khlebovich) dari Google Play Store.
   - **iOS / Android**: Atau gunakan aplikasi seperti **DroidCam** / **Iriun Webcam**.
2. **Jalankan server di HP**:
   - Sambungkan HP dan Laptop ke jaringan **Wi-Fi yang sama** (atau hotspot HP ke laptop).
   - Di aplikasi *IP Webcam*, scroll ke paling bawah dan pilih **"Start server"**.
   - Di layar HP akan muncul alamat URL, misalnya: `http://192.168.1.50:8080`
3. **Jalankan aplikasi dengan URL kamera HP**:
   ```powershell
   python app.py --source http://192.168.1.50:8080/video
   ```
   *(Ganti `192.168.1.50:8080` sesuai IP yang tertera di layar HP Anda, dan tambahkan `/video` di belakangnya)*.

---

## 👤 Cara Mendaftarkan Wajah (Face Enrollment)

Ada 3 cara mudah untuk mendaftarkan wajah orang baru ke sistem:

### Cara 1: Langsung dari Kamera yang Sedang Aktif (Paling Cepat)
1. Jalankan `python app.py`.
2. Hadapkan wajah orang yang ingin didaftarkan ke kamera.
3. Tekan tombol **`S`** pada keyboard.
4. Jendela dialog kecil akan muncul menanyakan nama. Ketikkan nama orang tersebut (misal: `Budi Santoso`), lalu klik **OK** / tekan **Enter**.
5. Wajah akan otomatis tersimpan ke folder `targets/Budi_Santoso.jpg` dan langsung dikenali di kamera tanpa perlu me-restart aplikasi!

---

### Cara 2: Menggunakan Skrip `register.py`
Anda bisa menggunakan skrip khusus untuk mendaftarkan wajah:

- **Menu Interaktif**:
  ```powershell
  python register.py
  ```
- **Ambil foto via webcam**:
  ```powershell
  python register.py --name "Budi Santoso"
  ```
  *(Arahkan wajah ke kotak hijau di layar lalu tekan **SPASI** untuk mengambil foto)*.
- **Ambil foto via kamera HP**:
  ```powershell
  python register.py --name "Siti Rahma" --source http://192.168.1.50:8080/video
  ```
- **Impor dari file foto yang sudah ada di laptop**:
  ```powershell
  python register.py --import "C:/Users/foto/budi.jpg" --name "Budi Santoso"
  ```
- **Melihat daftar semua orang yang sudah terdaftar**:
  ```powershell
  python register.py --list
  ```

---

### Cara 3: Salin Foto Manual ke Folder `targets/`
Anda juga bisa langsung menyalin file foto wajah ke dalam folder `targets/`.
- Nama file akan otomatis menjadi nama orang:
  - `targets/Jokowi.jpg` ➔ Dikenali sebagai **Jokowi**
  - `targets/Budi_Santoso.png` ➔ Dikenali sebagai **Budi Santoso**
- Jika aplikasi `app.py` sedang berjalan saat Anda menambahkan foto, cukup tekan tombol **`R`** pada keyboard untuk me-reload database seketika.

---

## ⌨️ Pintasan Keyboard (Saat Kamera Berjalan)

| Tombol | Fungsi |
| :---: | :--- |
| **`Q`** atau **`ESC`** | Keluar dari aplikasi dan menutup kamera. |
| **`S`** | Ambil foto wajah saat ini dan daftarkan nama baru. |
| **`R`** | Muat ulang (*reload*) database wajah dari folder `targets/`. |
| **`H`** | Sembunyikan / tampilkan banner HUD status di atas layar. |
| **`L`** | Sembunyikan / tampilkan 5 titik kunci wajah (*landmarks*). |

---

## ⚙️ Parameter & Konfigurasi Lanjutan

Anda dapat menyesuaikan parameter saat menjalankan `app.py`:

```powershell
python app.py [opsi...]
```

| Opsi | Default | Penjelasan |
| :--- | :---: | :--- |
| `--source` | `0` | Sumber kamera: angka `0` untuk webcam laptop, atau URL streaming HTTP/RTSP untuk kamera HP. |
| `--threshold` | `0.50` | Batas ambang Cosine Similarity (skala 0.0 - 1.0). Skor di atas threshold dikenali sebagai nama orang; di bawah threshold dianggap `Unknown`. |
| `--model` | `buffalo_sc` | Pilihan model InsightFace: `buffalo_sc` (ringan & cepat untuk CPU) atau `buffalo_l` (akurasi maksimal). |
| `--width` | `None` | Mengubah lebar resolusi video (misal: `--width 1280`). |
| `--height` | `None` | Mengubah tinggi resolusi video (misal: `--height 720`). |
| `--targets` | `targets` | Folder tempat penyimpanan foto referensi wajah. |
| `--skip-detection` | `0` | Lewati deteksi wajah tiap $N$ frame jika ingin lebih menghemat daya baterai/CPU laptop (misal `--skip-detection 1`). |
| `--gpu` | - | Mengaktifkan akselerasi CUDA GPU jika memiliki kartu grafis NVIDIA dan `onnxruntime-gpu`. |

### Contoh Penggunaan Kustom:
- **Tingkatkan ketelitian pencocokan wajah** (mencegah salah kenali orang lain):
  ```powershell
  python app.py --threshold 0.58
  ```
- **Kamera HP dengan resolusi diatur ke 720p**:
  ```powershell
  python app.py --source http://192.168.1.50:8080/video --width 1280 --height 720
  ```

---

## 💡 Tips & Solusi Masalah (Troubleshooting)

1. **Webcam laptop tidak mau terbuka (`cap.isOpened()` gagal)**:
   - Pastikan webcam tidak sedang dipakai aplikasi lain (seperti aplikasi Camera bawaan Windows, Zoom, Google Meet, atau Microsoft Teams).
   - Jika laptop Anda punya kamera eksternal USB, coba ganti sumber kamera dengan `--source 1`.
2. **Kamera HP lambat atau lag**:
   - Turunkan resolusi kamera di pengaturan aplikasi *IP Webcam* di HP (misalnya atur ke resolusi `1280x720` atau `960x540`).
   - Pastikan laptop dan HP terhubung ke frekuensi Wi-Fi yang stabil (disarankan Wi-Fi 5 GHz jika tersedia).
3. **Threshold Rekomendasi**:
   - Nilai **`0.45` - `0.55`** sangat ideal untuk mengenali wajah meski orang tersebut memakai kacamata atau gaya rambut berubah.
   - Nilai **`0.60` - `0.65`** cocok untuk sistem absensi/keamanan yang memerlukan verifikasi sangat ketat.
