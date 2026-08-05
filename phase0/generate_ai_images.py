"""
generate_ai_images.py
======================
Generates 300 AI images using stabilityai/sd-turbo (1-step diffusion).
SD-Turbo requires ~4GB VRAM and generates 512×512 images in ~0.2s each.
Uses the COCO captions as prompts so images are in-domain with real set.

Total generation time: ~60 seconds on RTX 3080Ti.
"""

import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import random

AI_DIR   = Path(__file__).parent / "data" / "ai"
AI_DIR.mkdir(parents=True, exist_ok=True)

# Diverse prompts aligned with COCO-style content (natural images)
PROMPTS = [
    "a photo of a dog in a park", "a cat sitting on a couch",
    "a bicycle parked on a street", "a bowl of fruit on a kitchen table",
    "people walking in a city", "a car on a road",
    "a bird flying in the sky", "a person cooking in a kitchen",
    "a child playing on a playground", "a boat on a lake",
    "a horse in a field", "a bus at a bus stop",
    "a woman reading a book", "a man eating pizza",
    "a group of people at a party", "a fire hydrant on a sidewalk",
    "a pizza on a table", "a baseball player hitting a ball",
    "a skier on a snowy slope", "a tennis player on a court",
    "a surfer on a wave", "a motorcycle on a highway",
    "a sandwich on a plate", "a hot dog at a fair",
    "a giraffe at a zoo", "an elephant in the wild",
    "a traffic light on a pole", "an airplane in the sky",
    "a train at a station", "a chair at a desk",
    "a laptop on a table", "a cell phone on a surface",
    "a vase of flowers on a shelf", "a potted plant in a room",
    "a refrigerator in a kitchen", "a toilet in a bathroom",
    "a bed in a bedroom", "a couch in a living room",
    "a TV on a stand", "a microwave in a kitchen",
    "a clock on a wall", "scissors on a table",
    "a teddy bear on a bed", "a hair dryer in a bathroom",
    "a toothbrush in a glass", "a book on a shelf",
    "a wine glass on a table", "a cup of coffee",
    "a bowl of soup", "a piece of cake on a plate",
    "a bakery with bread", "a supermarket aisle",
    "a restaurant interior", "a cafe on a street",
    "a park bench under trees", "a beach with waves",
    "a mountain lake", "a city skyline at night",
    "a suburban street", "a farm with crops",
]

def generate_images(n: int = 300, batch_size: int = 4,
                    device: str = "cuda"):
    print(f"[Generate] Loading SD-Turbo on {device} …")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[Generate] Model loaded. Starting generation …")

    existing = len(list(AI_DIR.glob("*.png")))
    if existing >= n:
        print(f"[Generate] Already have {existing} AI images. Done.")
        return

    generated = existing
    prompt_cycle = PROMPTS * ((n // len(PROMPTS)) + 2)
    random.seed(42)
    random.shuffle(prompt_cycle)

    from tqdm import tqdm
    pbar = tqdm(total=n - existing, desc="Generating AI images")

    i = 0
    while generated < n:
        batch_prompts = prompt_cycle[i:i+batch_size]
        i += batch_size
        if not batch_prompts:
            break
        try:
            images = pipe(
                prompt=batch_prompts,
                num_inference_steps=1,
                guidance_scale=0.0,
            ).images
            for img in images:
                if generated >= n:
                    break
                out = AI_DIR / f"sdturbo_{generated:05d}.png"
                img.save(out)
                generated += 1
                pbar.update(1)
        except Exception as e:
            print(f"[Generate] Error in batch: {e}")
            continue

    pbar.close()
    print(f"[Generate] Done. Generated {generated} AI images to {AI_DIR}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_images(n=300, batch_size=4, device=device)
    total = len(list(AI_DIR.glob("*.png")))
    print(f"[Generate] Total AI images in {AI_DIR}: {total}")
