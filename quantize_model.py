"""
Phase 3 - Step 3: INT8 Post-Training Quantization
Run INSIDE the Vitis-AI Docker container:
  conda activate vitis-ai-pytorch
  python quantize_model.py

Falls back to onnxruntime quantization if vai_q_onnx is unavailable.
"""

import os
import sys
import numpy as np
from calibration_reader import VisionEncoderCalibrationReader

# ─────────────────────────────────────────────
INPUT_MODEL   = "./vision_encoder.onnx"
OUTPUT_MODEL  = "./vision_encoder_int8.onnx"
CALIB_DIR     = "./calibration_dataset"
MAX_CALIB_IMG = 250
# ─────────────────────────────────────────────


def quantize_with_vitis_ai(reader):
    """
    Preferred path: use Vitis-AI vai_q_onnx for DPU-compatible INT8 model.
    Must be run inside Vitis-AI Docker container.
    """
    from vai_q_onnx import quantize_static, QuantType, CalibrationMethod

    print("[Vitis-AI] Running vai_q_onnx static quantization...")
    quantize_static(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        calibration_data_reader=reader,
        quant_format="QDQ",                     # QDQ format required for DPU
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,                        # per-channel weight quantization
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": False,
            "ForceQuantizeNoInputCheck": False,
        },
    )
    print("  ✅ Vitis-AI quantization complete")


def quantize_with_onnxruntime(reader):
    """
    Fallback path: use standard onnxruntime quantization.
    Produces INT8 ONNX model — may need re-compilation for DPU.
    """
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        QuantFormat,
        CalibrationMethod,
    )

    print("[onnxruntime] Running static INT8 quantization...")
    quantize_static(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    print("  ✅ onnxruntime quantization complete")


def verify_quantized_model():
    """Run a quick inference pass on the quantized model to verify it loads."""
    import onnxruntime as ort

    print("\n[Verification] Loading quantized model...")
    sess = ort.InferenceSession(OUTPUT_MODEL, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: dummy})
    print(f"  Output shape : {outputs[0].shape}")
    print(f"  Output dtype : {outputs[0].dtype}")
    print(f"  Output range : [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
    print("  ✅ Quantized model inference OK")


def compare_model_sizes():
    """Print FP32 vs INT8 model size comparison."""
    fp32_mb = os.path.getsize(INPUT_MODEL) / 1e6
    int8_mb = os.path.getsize(OUTPUT_MODEL) / 1e6
    ratio   = (1 - int8_mb / fp32_mb) * 100
    print(f"\n  FP32 model : {fp32_mb:7.1f} MB")
    print(f"  INT8 model : {int8_mb:7.1f} MB  ({ratio:.0f}% smaller)")
    print(f"  Compression: {fp32_mb / int8_mb:.2f}x")


def main():
    print("=" * 60)
    print("  INT8 Post-Training Quantization")
    print("=" * 60)

    if not os.path.exists(INPUT_MODEL):
        print(f"ERROR: Input model not found: {INPUT_MODEL}")
        print("Run export_onnx.py first.")
        sys.exit(1)

    if not os.path.exists(CALIB_DIR) or \
       not any(f.endswith((".jpg", ".png", ".jpeg")) for f in os.listdir(CALIB_DIR)):
        print(f"ERROR: Calibration dataset not found: {CALIB_DIR}")
        print("Run prepare_calibration_data.py first.")
        sys.exit(1)

    print(f"\n[Config]")
    print(f"  Input  : {INPUT_MODEL}")
    print(f"  Output : {OUTPUT_MODEL}")
    print(f"  Calib  : {CALIB_DIR}")

    # Build calibration reader
    reader = VisionEncoderCalibrationReader(
        calib_dir=CALIB_DIR,
        input_size=224,
        max_images=MAX_CALIB_IMG,
    )

    # Try Vitis-AI first, fall back to onnxruntime
    try:
        import vai_q_onnx  # noqa
        quantize_with_vitis_ai(reader)
    except ImportError:
        print("[INFO] vai_q_onnx not found — using onnxruntime quantization as fallback")
        print("       (For final DPU deployment, run this inside Vitis-AI Docker container)")
        quantize_with_onnxruntime(reader)

    # Post-quantization checks
    verify_quantized_model()
    compare_model_sizes()

    print(f"\n✅ Quantized model saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
