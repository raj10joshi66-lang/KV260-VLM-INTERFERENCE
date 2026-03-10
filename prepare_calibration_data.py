"""
Phase 3 - Step 1: Download and prepare calibration images from OCRBench
Run: python prepare_calibration_data.py
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
CALIB_DIR   = "./calibration_dataset"
N_IMAGES    = 250       # number of images to collect
IMAGE_SIZE  = 224
# ─────────────────────────────────────────────


def download_from_ocrbench():
    """Download calibration images from OCRBench dataset."""
    try:
        from datasets import load_dataset
        print("[OCRBench] Loading dataset from HuggingFace Hub...")
        print("           (First download may take 2-5 minutes...)")
        dataset = load_dataset("echo840/OCRBench", split="test")
        print(f"  ✓ OCRBench loaded: {len(dataset)} samples available")
        return dataset
    except Exception as e:
        print(f"  ✗ Could not load OCRBench: {e}")
        print("    (Will use synthetic fallback instead)")
        return None


def generate_synthetic_fallback(save_dir, n=250):
    """
    Generate synthetic calibration images if OCRBench is unavailable.
    These simulate text-on-background images typical of OCR tasks.
    """
    print("[FALLBACK] Generating synthetic calibration images...")
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        from PIL import ImageDraw
        ImageFont = None

    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    texts = ["Hello", "12345", "ABC", "Test", "OpenCV", "2024",
             "FPGA", "AMD", "VLM", "OCR", "Image", "Text", "KV260"]

    for i in range(n):
        bg_color = tuple(np.random.randint(180, 255, 3).tolist())
        fg_color = tuple(np.random.randint(0, 80, 3).tolist())
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=bg_color)
        draw = ImageDraw.Draw(img)
        text = texts[i % len(texts)] + str(i)
        draw.text((20, IMAGE_SIZE // 2 - 10), text, fill=fg_color)
        img.save(os.path.join(save_dir, f"synth_{i:04d}.jpg"), quality=90)
        saved += 1

    print(f"  Generated {saved} synthetic images -> {save_dir}")
    return saved


def preprocess_image(img, size=IMAGE_SIZE):
    """Resize + normalize image to tensor."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = img.convert("RGB").resize((size, size))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis, :]    # (1, 3, H, W)


def prepare_calibration_data():
    os.makedirs(CALIB_DIR, exist_ok=True)

    # ── Try to load from OCRBench ────────────────────────────────────
    dataset = download_from_ocrbench()
    saved   = 0
    meta    = []

    if dataset is not None:
        print(f"[1/2] Saving {N_IMAGES} calibration images to {CALIB_DIR}...")
        for idx, sample in enumerate(dataset):
            if saved >= N_IMAGES:
                break
            try:
                img = sample.get("image") or sample.get("img")
                q   = sample.get("question", "")
                a   = str(sample.get("answer", ""))

                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(np.array(img))

                fname = f"ocr_{saved:04d}.jpg"
                fpath = os.path.join(CALIB_DIR, fname)
                img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)).save(fpath, quality=90)
                meta.append({"file": fname, "question": q, "answer": a})
                saved += 1

                if saved % 50 == 0:
                    print(f"  Saved {saved}/{N_IMAGES} ...")
            except Exception as e:
                print(f"  Skipping sample {idx}: {e}")
                continue

        print(f"  Saved {saved} images from OCRBench.")
    else:
        saved = generate_synthetic_fallback(CALIB_DIR, N_IMAGES)

    # ── Save metadata ────────────────────────────────────────────────
    if meta:
        meta_path = os.path.join(CALIB_DIR, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved: {meta_path}")

    # ── Verify preprocessing pipeline ───────────────────────────────
    print("\n[2/2] Verifying preprocessing pipeline...")
    files = [f for f in os.listdir(CALIB_DIR) if f.endswith(".jpg")]
    sample_img = Image.open(os.path.join(CALIB_DIR, files[0]))
    tensor = preprocess_image(sample_img)
    print(f"  Sample image preprocessed: {tensor.shape}  "
          f"range=[{tensor.min():.3f}, {tensor.max():.3f}]")

    print(f"\n✅ Calibration dataset ready: {len(files)} images in {CALIB_DIR}")


if __name__ == "__main__":
    prepare_calibration_data()
