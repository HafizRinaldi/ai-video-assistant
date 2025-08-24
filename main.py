import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from core_logic import process_and_index_video
from qa_logic import answer_question_from_video

load_dotenv()

app = FastAPI(
    title="Asisten Video AI",
    description="Unggah video dan ajukan pertanyaan tentang isinya.",
    version="1.0.0"
)

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return "static/index.html"

@app.post("/index-video/", summary="Unggah dan Indeks Video")
async def index_video(file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Hanya file .mp4 yang didukung.")

    video_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        print(f"File berhasil disimpan di: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")

    try:
        print(f"Memulai proses indeksasi untuk video_id: {video_id}")
        process_and_index_video(video_path=file_path, video_id=video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proses indeksasi gagal: {e}")

    return {
        "message": "Video berhasil diunggah dan sedang diindeks.",
        "video_id": video_id,
        "filename": file.filename
    }

@app.get("/query-video/", summary="Ajukan Pertanyaan ke Video")
async def query_video(
    video_id: str = Query(..., description="ID unik dari video yang sudah diindeks."),
    question: str = Query(..., description="Pertanyaan Anda tentang isi video.")
):
    print(f"Menerima pertanyaan untuk video_id: {video_id}")
    print(f"Pertanyaan: {question}")
    
    if not os.path.exists(f"{video_id}.index"):
        raise HTTPException(status_code=404, detail=f"Video dengan ID '{video_id}' tidak ditemukan atau belum diindeks.")

    try:
        answer = answer_question_from_video(video_id=video_id, query=question)
        return {
            "video_id": video_id,
            "question": question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghasilkan jawaban: {e}")

# --- ENDPOINT BARU UNTUK DOWNLOAD TRANSKRIP ---
@app.get("/get-transcript/{video_id}", summary="Download Transkrip Lengkap")
async def get_transcript(video_id: str):
    """
    Menyediakan file transkrip .txt untuk diunduh.
    """
    transcript_path = f"{video_id}.txt"
    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="File transkrip tidak ditemukan.")
    
    return FileResponse(
        path=transcript_path, 
        filename=f"transcript_{video_id}.txt", 
        media_type='text/plain'
    )
