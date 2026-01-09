import os, tempfile, json
from pathlib import Path
from tqdm import tqdm
from media import extract_audio, sample_frames
from vision import caption_image
from llm import get_best_groq_model, summarize_chunk, final_summary
from text_utils import mmss, chunk_text, format_transcript
from models import whisper_model
from config import PROJECT_DIR

def run_pipeline(video_path):
    tmp = tempfile.mkdtemp(prefix="vid_")
    audio = os.path.join(tmp, "audio.wav")
    frames_dir = os.path.join(tmp, "frames")

    extract_audio(video_path, audio)
    transcript = format_transcript(
        whisper_model.transcribe(audio)["text"]
    )

    sample_frames(video_path, frames_dir)
    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))

    captions = []
    for i, frame in enumerate(tqdm(frames[:600])):
        captions.append({
            "time": mmss(i * 2),
            "caption": caption_image(str(frame))
        })

    model = get_best_groq_model()
    bullets = []
    for chunk in chunk_text(transcript):
        bullets.extend(summarize_chunk(model, chunk)["bullets"])

    return final_summary(model, bullets, captions), transcript
