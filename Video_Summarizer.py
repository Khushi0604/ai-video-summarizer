import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import torch
import whisper
from groq import Groq
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

# =============================
# CONFIG
# =============================
PROJECT_DIR = r"C:\Users\Khushi Nanwani\projects\AI Video summarizer"
os.makedirs(PROJECT_DIR, exist_ok=True)

GROQ_FREE_MODELS = ["llama-3.1-8b-instant", "mixtral-8x7b"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# MODEL LOADING
# =============================
print("Loading Whisper (small)...")
whisper_model = whisper.load_model("small")

print("Loading BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)


# =============================
# UTILITIES
# =============================
def get_best_groq_model():
    for model in GROQ_FREE_MODELS:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print(f"Using Groq model: {model}")
            return model
        except Exception:
            continue
    raise RuntimeError("No free Groq model available")


def extract_audio(video_path, out_audio):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            out_audio,
        ],
        check=True,
    )


def sample_frames(video_path, out_dir, fps=0.5):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-qscale:v",
            "2",
            os.path.join(out_dir, "frame_%05d.jpg"),
        ],
        check=True,
    )


def transcribe_audio(audio_path):
    return whisper_model.transcribe(audio_path)["text"]


def caption_image(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def mmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def chunk_text(text, max_chars=8000):
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


# =============================
# LLM CALLS
# =============================
def summarize_chunk(model, text, retries=3):
    prompt = f"""
Summarize this transcript chunk into concise bullet points.

TEXT:
{text}

Return JSON ONLY in this format:
{{"bullets":[ "...", "..."]}}
"""

    for attempt in range(retries):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You MUST return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        msg = resp.choices[0].message
        raw = msg.content.strip() if msg and msg.content else ""

        cleaned = clean_json(raw)

        if not cleaned:
            print("⚠ Empty response, retrying...")
            continue

        try:
            data = json.loads(cleaned)
            if isinstance(data.get("bullets"), list):
                return data
        except json.JSONDecodeError:
            print("⚠ Invalid JSON, retrying...")

    # FINAL fallback (never crash)
    print("⚠ Falling back to text bullets")
    bullets = [
        line.strip("-• ").strip() for line in raw.splitlines() if len(line.strip()) > 10
    ]
    return {"bullets": bullets[:8]}


def format_transcript(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence-wise line breaks
    text = re.sub(r"([.!?])\s+", r"\1\n", text)

    # Group sentences into readable paragraphs (6–8 lines)
    lines = text.split("\n")
    paragraphs = []
    buf = []

    for line in lines:
        if line.strip():
            buf.append(line.strip())
        if len(buf) >= 6:
            paragraphs.append(" ".join(buf))
            buf = []

    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(paragraphs)


def final_summary(model, bullets, frames, retries=3):
    frame_text = "\n".join([f"- {f['time']}: {f['caption']}" for f in frames])
    bullet_text = "\n".join(bullets)

    prompt = f"""
Create final video summary.

Requirements:
- 10-12 bullets
- 5–7 highlights with mm:ss timestamps

Bullets:
{bullet_text}

Frames:
{frame_text}

Return JSON ONLY:
{{"bullets":[...],"highlights":[{{"time":"mm:ss","note":"..."}}]}}
"""

    for _ in range(retries):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. No explanations.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        cleaned = clean_json(resp.choices[0].message.content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("⚠ Retrying final summary...")

    raise RuntimeError("Final summary failed after retries")


# =============================
# MAIN PIPELINE
# =============================
def main(video_path):
    tmp = tempfile.mkdtemp(prefix="vid_")
    audio = os.path.join(tmp, "audio.wav")
    frames_dir = os.path.join(tmp, "frames")

    base_name = Path(video_path).stem

    print("Extracting audio...")
    extract_audio(video_path, audio)

    print("Transcribing audio...")
    raw_transcript = transcribe_audio(audio)
    transcript = format_transcript(raw_transcript)

    transcript_path = os.path.join(PROJECT_DIR, f"{base_name}_FULL_TRANSCRIPT.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print("Sampling frames...")
    sample_frames(video_path, frames_dir)

    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))
    captions = []

    print(f"Captioning {len(frames)} frames...")
    MAX_FRAMES = 600

    for i, frame in enumerate(tqdm(frames[:MAX_FRAMES])):
        cap = caption_image(str(frame))
        captions.append({"time": mmss(i * 2), "caption": cap})

    model = get_best_groq_model()

    print("Chunking transcript...")
    chunks = chunk_text(transcript)

    print("Summarizing chunks...")
    all_bullets = []
    for chunk in chunks:
        result = summarize_chunk(model, chunk)
        all_bullets.extend(result["bullets"])

    print("Producing final summary...")
    final = final_summary(model, all_bullets, captions)

    base = Path(video_path).stem
    json_out = os.path.join(PROJECT_DIR, f"{base}_summary.json")
    md_out = os.path.join(PROJECT_DIR, f"{base}_SUMMARY_FOR_GITHUB.md")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    with open(md_out, "w", encoding="utf-8") as f:
        f.write(f"# Summary — {base}\n")
        f.write("**Automatically generated summary (transcript + visual frames)**\n\n")

        f.write("## Bullet Summary\n")
        for b in final["bullets"]:
            f.write(f"- {b}\n")

        f.write("\n## Highlights\n")
        for h in final["highlights"]:
            f.write(f"- **{h['time']}** — {h['note']}\n")

        f.write("\n---\n")
        f.write("**Transcript (trimmed):**\n\n")

        # Trim transcript for README (avoid wall of text)
        trimmed = transcript[:3000]
        f.write(trimmed)

    print("\nDONE")
    print("Saved:", json_out)
    print("Saved:", md_out)


# =============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_summarizer.py <video.mp4>")
        exit(1)
    main(sys.argv[1])
