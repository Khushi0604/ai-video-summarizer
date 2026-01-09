# AI Video Summarizer (Multimodal)
This project extracts audio and frames from a video, transcribes speech using Whisper,
generates image captions using BLIP, and produces a structured summary using Groq LLMs.

## Features
- Whisper-based transcription
- Automatic transcript formatting
- Frame captioning with BLIP
- Chunked LLM summarization (JSON-safe)
- GitHub-ready markdown output

## Setup
1. Clone repository
```bash
git clone https://github.com/your-username/ai-video-summarizer.git
cd ai-video-summarizer

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3.  Install dependencies
pip install -r requirements.txt

4. Set environment variables
setx GROQ_API_KEY "your_api_key_here"      #For Windows

export GROQ_API_KEY="your_api_key_here"    #For MacOS/Linux

## Usage
python video_summarizer.py input_video.mp4

## Notes

All AI models are used via APIs or open-source libraries.
Project emphasizes pipeline design, error handling, and prompt control.
AI assistance was used during development for debugging and iteration.

## Outputs
video name_ FULL_TRANSCRIPT.txt
video_name_summary.json
video_name_SUMMARY_FOR_GITHUB.md

## Tested on
Windows 10
Python 3.10+
CPU & CUDA

## 🔑 API Key Setup

This project uses an LLM API (Groq / OpenAI-compatible) for summarization.

### Step-by-step to get your API key

1. Go to https://console.groq.com/
2. Sign in using Google/GitHub or create an account
3. Navigate to **API Keys**
4. Click **Create API Key**
5. Copy the generated key

### How to use the API key locally

⚠️ **Do NOT hardcode your API key in the code**

#### Windows (PowerShell)
```powershell
setx GROQ_API_KEY "your_api_key_here"

# For MacOS/Linux
export GROQ_API_KEY="your_api_key_here"


## 📁 Sample Outputs
Check the `samples/` folder for:
- Demo input video
- Generated markdown summary
- Transcript snippet

These are included to help reviewers and recruiters quickly understand the project output without running the code.
