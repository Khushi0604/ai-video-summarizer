import subprocess
import os

def extract_audio(video_path, out_audio):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_audio],
        check=True,
    )

def sample_frames(video_path, out_dir, fps=0.5):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps={fps}", "-qscale:v", "2",
         os.path.join(out_dir, "frame_%05d.jpg")],
        check=True,
    )
