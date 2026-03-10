"""
Phase 3 - Step 2: Calibration Data Reader for Vitis-AI / ONNX Runtime quantization
Used by quantize_model.py — not run directly.
"""

import os
import glob
import numpy as np
from PIL import Image


class VisionEncoderCalibrationReader:
    """
    Calibration data reader compatible with:
      - onnxruntime.quantization.CalibrationDataReader  (host quantization)
      - Vitis-AI vai_q_onnx calibration interface        (inside Docker)

    Yields one preprocessed image tensor at a time.
    """

    # ImageNet normalization constants (used by SigLIP / ViT-based encoders)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, calib_dir: str, input_size: int = 224, max_images: int = 250):
        self.image_paths = sorted(
            glob.glob(os.path.join(calib_dir, "*.jpg")) +
            glob.glob(os.path.join(calib_dir, "*.png")) +
            glob.glob(os.path.join(calib_dir, "*.jpeg"))
        )[:max_images]

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in calibration directory: {calib_dir}\n"
                f"Run prepare_calibration_data.py first."
            )

        self.input_size = input_size
        self.idx        = 0
        self.input_name = "input"   # must match ONNX export input name
        print(f"[CalibReader] Loaded {len(self.image_paths)} calibration images "
              f"from {calib_dir}")

    # ── Core interface ──────────────────────────────────────────────
    def get_next(self):
        """Return next calibration batch or None when exhausted."""
        if self.idx >= len(self.image_paths):
            return None
        tensor = self._preprocess(self.image_paths[self.idx])
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"  Calibrated {self.idx}/{len(self.image_paths)} images...")
        return {self.input_name: tensor}

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result

    def __len__(self):
        return len(self.image_paths)

    # ── Preprocessing ───────────────────────────────────────────────
    def _preprocess(self, img_path: str) -> np.ndarray:
        """
        Preprocessing pipeline:
          1. Load RGB image
          2. Resize to (input_size x input_size)
          3. Normalize: (pixel/255 - mean) / std
          4. Transpose HWC -> CHW
          5. Add batch dimension -> (1, 3, H, W)
        """
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)      # HWC -> CHW
        return arr[np.newaxis, :]          # -> (1, 3, H, W)

    def rewind(self):
        """Reset iterator to beginning (allows multiple calibration passes)."""
        self.idx = 0


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    calib_dir = sys.argv[1] if len(sys.argv) > 1 else "./calibration_dataset"
    reader = VisionEncoderCalibrationReader(calib_dir)
    batch  = reader.get_next()
    if batch:
        arr = list(batch.values())[0]
        print(f"Batch shape : {arr.shape}")
        print(f"Batch dtype : {arr.dtype}")
        print(f"Value range : [{arr.min():.4f}, {arr.max():.4f}]")
        print("✅ CalibrationReader OK")
