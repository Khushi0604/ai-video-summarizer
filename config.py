import os
import torch
from groq import Groq

PROJECT_DIR = r"C:\Users\Khushi Nanwani\projects\AI Video summarizer"
os.makedirs(PROJECT_DIR, exist_ok=True)

GROQ_FREE_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"
