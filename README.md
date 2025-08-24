# 🤖 Asisten Video AI (Versi RAG Lokal)

Proyek ini adalah asisten AI canggih yang berjalan sepenuhnya di komputer Anda (self-hosted). Aplikasi ini dapat mengindeks konten video dan menjawab pertanyaan dalam bahasa alami tentang isinya. Proyek ini dibangun menggunakan arsitektur **Retrieval-Augmented Generation (RAG)**, dengan memanfaatkan model open-source canggih dari Hugging Face, termasuk **Large Language Model (LLM) Llama 3**, untuk menyediakan pengalaman tanya jawab yang sepenuhnya pribadi, gratis, dan dipercepat oleh GPU.

Aplikasi ini memiliki antarmuka web modern yang dibangun dengan **FastAPI** dan **Tailwind CSS**, memungkinkan pengguna untuk dengan mudah mengunggah video, melihat transkrip lengkap, dan berinteraksi dengan AI.

---

## ✨ Fitur

* **Indeksasi Video**: Memproses file video `.mp4` apa pun untuk membuat indeks semantik yang dapat dicari dari konten yang diucapkan.
* **Tanya Jawab Cerdas**: Mengajukan pertanyaan dalam bahasa biasa tentang konten video dan menerima jawaban yang akurat dan sadar konteks yang dihasilkan oleh Large Language Model.
* **UI Interaktif**: Antarmuka web bergaya dasbor yang bersih untuk mengunggah video, melihat transkrip, dan berinteraksi dengan AI.
* **Tampilan & Unduh Transkrip Lengkap**: Melihat transkrip lengkap langsung di UI atau mengunduhnya sebagai file `.txt`.
* **Akselerasi GPU**: Secara otomatis memanfaatkan GPU NVIDIA melalui **PyTorch** dan **CUDA** untuk transkripsi dan pembuatan jawaban yang jauh lebih cepat.
* **100% Lokal & Pribadi**: Berjalan sepenuhnya di mesin lokal Anda. Video dan data Anda tidak pernah dikirim ke layanan pihak ketiga, memastikan privasi lengkap dan tanpa biaya API.

---

## 📸 Screenshot

Berikut adalah pratinjau antarmuka pengguna aplikasi:

![Pratinjau antarmuka pengguna aplikasi](https://raw.githubusercontent.com/NAMA_ANDA/asisten-video-ai/main/screenshots/dashboard.png)

---

## 🛠️ Tumpukan Teknologi

Proyek ini menggabungkan beberapa teknologi canggih dari ekosistem AI Python.

* **Backend**: FastAPI, Uvicorn
* **Frontend**: HTML, Tailwind CSS, JavaScript
* **Model AI (Hugging Face)**:
    * **Transkripsi**: `openai/whisper-base` untuk mengubah ucapan menjadi teks.
    * **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` untuk membuat representasi vektor semantik dari teks.
    * **AI Generatif (LLM)**: `meta-llama/Meta-Llama-3-8B-Instruct` untuk memahami konteks dan menghasilkan jawaban.
* **Library Inti AI**: `transformers`, `torch`, `sentence-transformers`, `accelerate`
* **Database Vektor**: `faiss-cpu` / `faiss-gpu` untuk pencarian kemiripan yang efisien.
* **Pemrosesan Video/Audio**: `ffmpeg-python`

---

## 🚀 Pengaturan dan Instalasi

Ikuti langkah-langkah ini untuk menjalankan proyek di mesin lokal Anda.

### Prasyarat

* Python 3.9+
* GPU NVIDIA dengan CUDA terinstal sangat direkomendasikan untuk performa yang baik.
* FFMPEG terinstal dan ditambahkan ke `PATH` environment variable sistem Anda.

### Langkah-langkah Instalasi

1.  Clone repositori ini:

    ```bash
    git clone [https://github.com/NAMA_ANDA/asisten-video-ai.git](https://github.com/NAMA_ANDA/asisten-video-ai.git)
    cd asisten-video-ai
    ```

2.  Buat dan aktifkan lingkungan virtual:

    ```bash
    # Membuat environment
    python -m venv venv
    
    # Aktivasi di Windows
    venv\Scripts\activate
    
    # Aktivasi di macOS/Linux
    source venv/bin/activate
    ```

3.  Instal semua library yang dibutuhkan:
    Pertama, instal PyTorch dengan dukungan CUDA. Kunjungi situs web PyTorch untuk mendapatkan perintah yang benar untuk versi CUDA Anda. Contoh:

    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

    Kemudian, instal paket lainnya dari file `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4.  Siapkan environment variable Anda:
    Buat file di direktori utama bernama `.env`.
    Tambahkan Token Akses Hugging Face Anda ke dalamnya. Anda memerlukan ini untuk mengunduh model Llama 3.

    ```
    HUGGING_FACE_TOKEN="hf_..."
    ```

5.  Login ke Hugging Face:
    Jalankan perintah ini di terminal Anda dan tempelkan token Anda saat diminta.

    ```bash
    huggingface-cli login
    ```

---

## ⚙️ Cara Menggunakan

1.  **Jalankan Server**
    Mulai server FastAPI dari terminal Anda (pastikan lingkungan virtual Anda aktif):

    ```bash
    uvicorn main:app --reload
    ```

2.  **Buka Aplikasi**
    Buka browser web Anda dan kunjungi `http://127.0.0.1:8000`.

3.  **Indeks sebuah Video**
    * Di UI web, klik "Choose File" dan pilih video `.mp4`.
    * Klik tombol "Unggah & Indeks Video".
    * Tunggu proses pengindeksan selesai. Ini bisa memakan waktu lama tergantung pada panjang video dan perangkat keras Anda. Transkrip lengkap akan muncul di panel kanan setelah selesai.

4.  **Ajukan Pertanyaan**
    * Setelah pengindeksan selesai, kotak input pertanyaan akan muncul.
    * Ketik pertanyaan Anda tentang konten video dan klik "Tanyakan pada AI".
    * Jawaban dari AI akan muncul di bawah kotak input.
