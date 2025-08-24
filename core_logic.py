import ffmpeg
import torch
import json
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --- FUNGSI 1: EKSTRAK AUDIO DARI VIDEO ---
def extract_audio_from_video(video_path: str, audio_path: str = "temp_audio.wav"):
    """
    Mengekstrak audio dari file video menggunakan ffmpeg dan menyimpannya sebagai file .wav.
    """
    print(f"Mengekstrak audio dari {video_path}...")
    try:
        ffmpeg.input(video_path).output(
            audio_path,
            acodec="pcm_s16le",
            ac=1,
            ar="16000"
        ).run(overwrite_output=True, quiet=True)
        print(f"Audio berhasil diekstrak ke {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"Error saat mengekstrak audio: {e.stderr.decode()}")
        return None

# --- FUNGSI 2: TRANSKRIPSI DENGAN TIMESTAMP ---
def transcribe_with_timestamps(audio_path: str):
    """
    Mentranskripsikan audio menggunakan Whisper dan menghasilkan chunk dengan stempel waktu
    serta teks lengkap.
    """
    print("Memulai transkripsi dengan stempel waktu...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device
    )
    
    result = pipe(audio_path, return_timestamps="word", chunk_length_s=30, stride_length_s=5)
    print("Transkripsi selesai.")
    # Kembalikan chunks untuk RAG dan teks lengkap untuk diunduh
    return result['chunks'], result['text']

# --- FUNGSI 3: BUAT POTONGAN TEKS (CHUNKS) ---
def create_text_chunks(transcript_chunks, chunk_duration_seconds=60):
    """
    Mengelompokkan hasil transkripsi kata-demi-kata menjadi potongan teks yang lebih besar.
    """
    print("Membuat potongan teks (chunks)...")
    chunks = []
    current_chunk = {"text": "", "start": None, "end": None}

    for word_data in transcript_chunks:
        text = word_data['text']
        start, end = word_data['timestamp']

        if current_chunk["start"] is None:
            current_chunk["start"] = start

        if end is not None and (end - current_chunk["start"]) > chunk_duration_seconds:
            current_chunk["end"] = current_chunk_text_end
            chunks.append(current_chunk)
            current_chunk = {"text": text, "start": start, "end": end}
        else:
            current_chunk["text"] += text
            current_chunk_text_end = end

    if current_chunk["text"]:
        current_chunk["end"] = current_chunk_text_end
        chunks.append(current_chunk)
        
    print(f"Berhasil membuat {len(chunks)} potongan teks.")
    return chunks

# --- FUNGSI 4: BUAT & SIMPAN EMBEDDINGS ---
def create_and_save_embeddings(text_chunks, video_id: str):
    """
    Membuat vector embeddings dan menyimpannya ke dalam indeks FAISS.
    """
    print("Membuat vector embeddings...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
    texts_to_embed = [chunk['text'] for chunk in text_chunks]
    
    embeddings = model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().numpy()
    
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    
    faiss.write_index(index, f"{video_id}.index")
    with open(f"{video_id}.json", "w") as f:
        json.dump(text_chunks, f)
        
    print(f"Embeddings dan data berhasil disimpan untuk video_id: {video_id}")

# --- FUNGSI UTAMA UNTUK MENGGABUNGKAN SEMUA ---
def process_and_index_video(video_path: str, video_id: str):
    """
    Menjalankan seluruh pipeline: ekstrak, transkripsi, chunking, dan embedding.
    """
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        return

    # Dapatkan chunks dan teks lengkap
    transcript_chunks, full_transcript = transcribe_with_timestamps(audio_path)
    
    # Simpan transkrip lengkap ke file .txt
    with open(f"{video_id}.txt", "w", encoding="utf-8") as f:
        f.write(full_transcript)
    print(f"Transkrip lengkap disimpan ke {video_id}.txt")
    
    text_chunks = create_text_chunks(transcript_chunks)
    
    create_and_save_embeddings(text_chunks, video_id)
    
    print(f"\n🎉 Video '{video_id}' berhasil diindeks!")

