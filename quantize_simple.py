"""
Alternative Phase 3.3 - Simplified Quantization (for testing)
For production: Use Vitis-AI Docker container with proper quantization

This creates a basic INT8 model without full static quantization.
Final deployment should use: quantize_model.py inside Vitis-AI Docker
"""

import os
import sys
import numpy as np
import onnx
from pathlib import Path

INPUT_MODEL = "./vision_encoder.onnx"
OUTPUT_MODEL = "./vision_encoder_int8_simple.onnx"


def fix_onnx_types():
    """
    Load ONNX model and verify it can be loaded.
    Returns True if model is valid.
    """
    try:
        model = onnx.load(INPUT_MODEL)
        print(f"✅ ONNX model loaded: {INPUT_MODEL}")
        print(f"   Model inputs: {[inp.name for inp in model.graph.input]}")
        print(f"   Model outputs: {[out.name for out in model.graph.output]}")
        return model
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return None


def quantize_with_dynamic_quantization():
    """
    Use ONNX Runtime's dynamic quantization as a fast approximation.
    This is NOT recommended for production — use Vitis-AI Docker instead.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print("\n[Dynamic Quantization] Running fast INT8 conversion...")
    print("  WARNING: This is a simplified approach for testing only.")
    print("  For production deployment, use Vitis-AI Docker quantization.")
    
    try:
        quantize_dynamic(
            INPUT_MODEL,
            OUTPUT_MODEL,
            weight_type=QuantType.QInt8,
        )
        print("  ✅ Dynamic quantization complete")
        return True
    except Exception as e:
        print(f"  ❌ Dynamic quantization failed: {e}")
        return False


def verify_models():
    """Compare FP32 and INT8 models."""
    import onnxruntime as ort
    
    print("\n[Verification] Loading and testing models...")
    
    if not os.path.exists(OUTPUT_MODEL):
        print(f"  ❌ Quantized model not found: {OUTPUT_MODEL}")
        return False
    
    # Test FP32 model
    print("\n  FP32 Model Test:")
    fp32_sess = ort.InferenceSession(INPUT_MODEL, providers=["CPUExecutionProvider"])
    fp32_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    fp32_input_name = fp32_sess.get_inputs()[0].name
    fp32_out = fp32_sess.run(None, {fp32_input_name: fp32_input})[0]
    print(f"    Shape: {fp32_out.shape}, dtype: {fp32_out.dtype}")
    print(f"    Range: [{fp32_out.min():.4f}, {fp32_out.max():.4f}]")
    
    # Test INT8 model
    print("\n  INT8 Model Test:")
    try:
        int8_sess = ort.InferenceSession(OUTPUT_MODEL, providers=["CPUExecutionProvider"])
        int8_input_name = int8_sess.get_inputs()[0].name
        int8_out = int8_sess.run(None, {int8_input_name: fp32_input})[0]
        print(f"    Shape: {int8_out.shape}, dtype: {int8_out.dtype}")
        print(f"    Range: [{int8_out.min():.4f}, {int8_out.max():.4f}]")
        
        # Compare sizes
        fp32_mb = os.path.getsize(INPUT_MODEL) / 1e6
        int8_mb = os.path.getsize(OUTPUT_MODEL) / 1e6
        ratio = (1 - int8_mb / fp32_mb) * 100
        print(f"\n  Model Sizes:")
        print(f"    FP32: {fp32_mb:.1f} MB")
        print(f"    INT8: {int8_mb:.1f} MB ({ratio:.0f}% smaller)")
        print(f"    Compression ratio: {fp32_mb / int8_mb:.2f}x")
        
        return True
    except Exception as e:
        print(f"    ❌ INT8 inference failed: {e}")
        return False


def main():
    print("=" * 70)
    print("  Simplified INT8 Quantization (Testing Only)")
    print("=" * 70)
    print("\n  ⚠️  For production KV260 deployment:")
    print("      1. Use Vitis-AI Docker container")
    print("      2. Run: quantize_model.py (with vai_q_onnx)")
    print("      3. This ensures DPU-compatible INT8 format\n")
    
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Input model not found: {INPUT_MODEL}")
        sys.exit(1)
    
    # Load and check model
    model = fix_onnx_types()
    if not model:
        sys.exit(1)
    
    # Run dynamic quantization
    if not quantize_with_dynamic_quantization():
        print("\n⚠️  Dynamic quantization failed or unavailable")
        print("    For production, use Vitis-AI Docker container")
        sys.exit(1)
    
    # Verify results
    if verify_models():
        print(f"\n✅ Quantized model ready: {OUTPUT_MODEL}")
        print("\n📌 NEXT STEPS FOR PRODUCTION:")
        print("   1. docker pull xilinx/vitis-ai-pytorch:latest")
        print("   2. docker run -it -v $PWD:/workspace xilinx/vitis-ai-pytorch:latest")
        print("   3. cd /workspace && python quantize_model.py")
        print("   4. This produces DPU-optimized vision_encoder_int8.onnx")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
