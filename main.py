import sys, json
from pathlib import Path
from pipeline import run_pipeline
from config import PROJECT_DIR

# CONFIG
# MODEL LOADING
# UTILITIES
# LLM CALLS

# MAIN PIPELINE
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video.mp4>")
        exit(1)

    video = sys.argv[1]
    base = Path(video).stem

    final, transcript = run_pipeline(video)

    with open(f"{PROJECT_DIR}/{base}_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    print("DONE")

