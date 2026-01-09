import json
from config import client, GROQ_FREE_MODELS
from text_utils import clean_json

def get_best_groq_model():
    for model in GROQ_FREE_MODELS:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return model
        except Exception:
            continue
    raise RuntimeError("No free Groq model available")

def summarize_chunk(model, text):
    prompt = f"""Summarize this transcript chunk into bullets.
Return JSON ONLY: {{"bullets":[...]}}"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt + text}],
    )

    return json.loads(clean_json(resp.choices[0].message.content))

def final_summary(model, bullets, frames):
    prompt = "Create final summary JSON only."
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(clean_json(resp.choices[0].message.content))
