from config import device
import whisper
from transformers import BlipForConditionalGeneration, BlipProcessor

print("Loading Whisper (small)...")
whisper_model = whisper.load_model("small")

print("Loading BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

