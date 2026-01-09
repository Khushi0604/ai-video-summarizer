import re
import json

def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

def chunk_text(text, max_chars=8000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()

def format_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])\s+", r"\1\n", text)

    lines = text.split("\n")
    paragraphs, buf = [], []

    for line in lines:
        if line.strip():
            buf.append(line.strip())
        if len(buf) >= 6:
            paragraphs.append(" ".join(buf))
            buf = []

    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(paragraphs)
