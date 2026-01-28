import os
import time
import random
from datetime import datetime

from google import genai
from google.genai import types

OUTPUT_DIR = "./generated_images"
MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro

IMAGE_SIZE = "1K"
ASPECT_RATIO = "4:3"  

BASE_PROMPT = """
A realistic dataset image for an autonomous search-and-rescue robot training.

A person lying on the ground in an indoor post-earthquake environment, WITHOUT any visible injuries.
No blood, no wounds, no violence, no medical equipment, no emergency responders.

Camera:
Ground-level camera positioned approximately 5 centimeters above the floor.
Low-angle perspective, like a rescue robot.
Wide-angle lens, natural look.

Environment:
Collapsed building interior, concrete floor, dust, small debris, broken tiles.
Soft natural lighting.

Pose:
Person lying naturally on the ground, full body visible in frame.
Arms and legs relaxed.

Style:
Photorealistic, dataset-style, neutral, no cinematic effects.
"""

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_images_from_response(response, out_path_prefix: str):
    saved = 0
    if response.parts is None:
        return saved
    for part in response.parts:
        if part.inline_data is not None:
            img = part.as_image()
            filename = f"{out_path_prefix}_{saved:02d}.png"
            img.save(filename)
            saved += 1
    return saved

def generate_one(client: genai.Client, prompt: str, out_dir: str, idx: int):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(out_dir, f"img_{idx:06d}_{ts}")

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=ASPECT_RATIO,
                image_size=IMAGE_SIZE,
            ),
        ),
    )

    n = save_images_from_response(response, prefix)

    meta_path = f"{prefix}_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(prompt.strip() + "\n")
        f.write(f"\nmodel={MODEL}\naspect_ratio={ASPECT_RATIO}\nimage_size={IMAGE_SIZE}\n")

    return n, meta_path

def main(total_images: int = 50, min_wait: int = 10, max_wait: int = 20, start_index: int = 0):
    ensure_dir(OUTPUT_DIR)
    client = genai.Client() 

    i = start_index
    generated = 0

    while generated < total_images:
        # basit çeşitlilik için küçük varyasyonlar
        variant = random.choice([
            "Person lying on back (supine).",
            "Person lying on side.",
            "Person partially covered with dust (no injury).",
            "Person wearing casual clothes.",
        ])
        prompt = BASE_PROMPT.strip() + "\n\nVariant:\n" + variant

        try:
            n, meta = generate_one(client, prompt, OUTPUT_DIR, i)
            if n == 0:
                print(f"[WARN] idx={i} no images returned, skipping")
                i += 1
                continue
            print(f"[OK] idx={i} saved_images={n} meta={meta}")
            generated += max(n, 1)
            i += 1

        except Exception as e:
            wait = random.randint(15, 30)
            print(f"[ERR] idx={i} error={e} -> retry in {wait}s")
            time.sleep(wait)
            continue

        wait = random.randint(min_wait, max_wait)
        time.sleep(wait)

if __name__ == "__main__":
    main(total_images=100, min_wait=5, max_wait=10, start_index=0)
