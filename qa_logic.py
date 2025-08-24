import faiss
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- FUNGSI 1: MUAT INDEKS DAN DATA ---
def load_index_and_data(video_id: str):
    """
    Memuat indeks FAISS dan file data JSON yang sesuai.
    """
    print(f"Memuat data untuk video_id: {video_id}...")
    try:
        index = faiss.read_index(f"{video_id}.index")
        with open(f"{video_id}.json", "r") as f:
            text_chunks = json.load(f)
        print("Data berhasil dimuat.")
        return index, text_chunks
    except FileNotFoundError:
        print(f"Error: File indeks atau data untuk '{video_id}' tidak ditemukan.")
        return None, None

# --- FUNGSI 2: CARI KONTEKS YANG RELEVAN ---
def find_relevant_context(query: str, index, text_chunks, top_k=3):
    """
    Mencari potongan teks yang paling relevan dengan pertanyaan pengguna di dalam indeks FAISS.
    """
    print(f"Mencari konteks untuk pertanyaan: '{query}'...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
    # Buat embedding untuk pertanyaan
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    
    # Lakukan pencarian di indeks FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Kumpulkan potongan teks yang relevan
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    
    print(f"Menemukan {len(relevant_chunks)} konteks yang relevan.")
    return relevant_chunks

# --- FUNGSI 3: BUAT JAWABAN DENGAN LLM ---
def generate_answer_with_llm(query: str, context_chunks):
    """
    Menggunakan LLM (Llama 3) untuk menghasilkan jawaban berdasarkan pertanyaan dan konteks yang ditemukan.
    """
    print("Menghasilkan jawaban dengan LLM...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Format konteks menjadi satu string yang mudah dibaca
    context_str = ""
    for chunk in context_chunks:
        start_time = int(chunk['start'])
        context_str += f"- (Menit {start_time // 60}:{start_time % 60:02d}) {chunk['text']}\n"

    # Template prompt untuk LLM
    prompt = f"""
    Anda adalah asisten AI yang membantu pengguna menemukan informasi di dalam video.
    Jawab pertanyaan pengguna HANYA berdasarkan konteks yang diberikan di bawah ini.
    Sebutkan stempel waktu (timestamp) jika relevan untuk membantu pengguna menemukan momen tersebut di video.

    KONTEKS DARI VIDEO:
    {context_str}

    PERTANYAAN PENGGUNA:
    {query}

    JAWABAN:
    """

    messages = [
        {"role": "system", "content": "Anda adalah asisten AI yang membantu pengguna menemukan informasi di dalam video."},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    
    print("Jawaban berhasil dibuat.")
    return answer

# --- FUNGSI UTAMA UNTUK MENGGABUNGKAN SEMUA ---
def answer_question_from_video(video_id: str, query: str):
    """
    Menjalankan seluruh pipeline tanya jawab.
    """
    # 1. Muat data
    index, text_chunks = load_index_and_data(video_id)
    if index is None:
        return "Gagal memuat data video. Pastikan video sudah diindeks."

    # 2. Cari konteks
    relevant_context = find_relevant_context(query, index, text_chunks)
    
    # 3. Buat jawaban
    answer = generate_answer_with_llm(query, relevant_context)
    
    return answer

# --- BLOK UNTUK PENGUJIAN LANGSUNG ---
if __name__ == "__main__":
    video_id_to_query = "sample_video_1"
    # Ganti dengan pertanyaan yang relevan dengan isi video Anda
    question = "teknik apa yang dijelaskan oleh pengajar?" 
    
    print(f"===== MENJAWAB PERTANYAAN UNTUK VIDEO '{video_id_to_query}' =====")
    final_answer = answer_question_from_video(video_id_to_query, question)
    print("\n--- JAWABAN FINAL ---")
    print(final_answer)
    print("======================")