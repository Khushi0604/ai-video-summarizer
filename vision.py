from PIL import Image
from models import blip_model, processor
from config import device

def caption_image(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)
