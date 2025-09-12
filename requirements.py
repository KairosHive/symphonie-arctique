# ---- Core numeric/audio ----
numpy>=1.23
scipy>=1.10
librosa>=0.10.2.post1
soundfile>=0.12.1
audioread>=3.0.0
tqdm>=4.66

# ---- Transformers / LLMs (CLAP + Gemma wrappers) ----
transformers>=4.45.0
accelerate>=0.31.0
safetensors>=0.4.3
tokenizers>=0.19.0
huggingface-hub>=0.22.0

# ---- Optional but handy ----
python-dotenv>=1.0.1


# 2) Install PyTorch build for your system (choose ONE)
#    CUDA 12.x (Linux/Windows, NVIDIA GPU):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch