import torch

if torch.cuda.is_available():
    print("✅ Sukses! PyTorch bisa melihat GPU Anda.")
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
    print("Model AI akan berjalan di GPU.")
else:
    print("❌ Gagal. PyTorch tidak bisa melihat GPU Anda.")
    print("Pastikan Anda telah menginstal PyTorch versi CUDA dan driver NVIDIA Anda sudah terbaru.")
    print("Model AI akan berjalan di CPU yang jauh lebih lambat.")