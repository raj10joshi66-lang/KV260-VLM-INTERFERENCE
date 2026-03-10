import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Try to import DPU runtime (available on KV260)
try:
    from vart import Graph, Runner
    HAS_DPU = True
    print("✓ DPU Runtime available")
except ImportError:
    HAS_DPU = False
    print("⚠ DPU Runtime not available (CPU fallback)")

try:
    import onnxruntime as ort
    HAS_ONNX = True
    print("✓ ONNX Runtime available")
except ImportError:
    HAS_ONNX = False


# ─── Configuration ───────────────────────────────────────────
XMODEL_PATH = "./compiled_model/vision_encoder.xmodel"
ONNX_PATH = "./models/vision_encoder_int8.onnx"
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DPUInference:
    """DPU inference wrapper for vision encoder."""
    
    def __init__(self, xmodel_path):
        """Load DPU model."""
        if not os.path.exists(xmodel_path):
            raise FileNotFoundError(f"xmodel not found: {xmodel_path}")
        
        self.graph = Graph.deserialize(xmodel_path)
        self.runner = self.graph.create_runner("normal")
        
        # Get input/output names
        input_tensor_list = self.runner.get_input_tensors()
        output_tensor_list = self.runner.get_output_tensors()
        
        self.input_name = input_tensor_list[0].name
        self.output_name = output_tensor_list[0].name
        
        print(f"  Input : {self.input_name}")
        print(f"  Output: {self.output_name}")
    
    def __call__(self, img_array):
        """Run inference on DPU."""
        # img_array should be (1, 3, 224, 224) float32
        job_id = self.runner.execute_async([img_array])
        self.runner.wait(job_id)
        outputs = self.runner.get_output(job_id)
        return outputs[0]  # Return first (and only) output


class ONNXInference:
    """ONNX Runtime inference fallback."""
    
    def __init__(self, onnx_path):
        """Load ONNX model."""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"  Input : {self.input_name}")
        print(f"  Output: {self.output_name}")
    
    def __call__(self, img_array):
        """Run inference on CPU."""
        output = self.session.run(
            [self.output_name],
            {self.input_name: img_array}
        )
        return output[0]


def preprocess_image(img_path: str) -> np.ndarray:
    """Preprocess image to (1, 3, 224, 224) float32."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, :]  # Add batch → (1, 3, 224, 224)


def benchmark_inference(runner, num_runs=10):
    """Benchmark inference latency."""
    print(f"\n[Benchmark] Running {num_runs} inference passes...")
    
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    _ = runner(dummy_input)
    
    # Time runs
    times = []
    for i in range(num_runs):
        t0 = time.time()
        output = runner(dummy_input)
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # Convert to ms
        
        if (i + 1) % max(1, num_runs // 3) == 0:
            print(f"  Run {i + 1}/{num_runs}: {times[-1]:.2f} ms")
    
    times = np.array(times)
    print(f"\nResults:")
    print(f"  Mean   : {times.mean():.2f} ms")
    print(f"  Median : {np.median(times):.2f} ms")
    print(f"  Stdev  : {times.std():.2f} ms")
    print(f"  Min    : {times.min():.2f} ms")
    print(f"  Max    : {times.max():.2f} ms")
    print(f"  Throughput: {1000 / times.mean():.1f} images/sec")


def test_single_image(runner, img_path):
    """Test inference on a single image."""
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found: {img_path}")
        return
    
    print(f"\n[Inference] Testing image: {img_path}")
    
    img = preprocess_image(img_path)
    print(f"  Input shape: {img.shape}")
    print(f"  Input range: [{img.min():.3f}, {img.max():.3f}]")
    
    t0 = time.time()
    output = runner(img)
    t1 = time.time()
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Latency: {(t1 - t0) * 1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="DPU Inference Test for SmolVLM Vision Encoder")
    parser.add_argument("--image", help="Test image path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on OCRBench")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--use-onnx", action="store_true", help="Force ONNX Runtime (skip DPU)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("  SmolVLM Vision Encoder - DPU Inference Test")
    print("="*60)
    print()
    
    # Select runtime
    runner = None
    runtime_name = None
    
    if not args.use_onnx and HAS_DPU:
        print("[Runtime] Loading DPU inference...")
        try:
            runner = DPUInference(XMODEL_PATH)
            runtime_name = "DPU"
        except Exception as e:
            print(f"  Failed: {e}")
            if HAS_ONNX:
                print("  Falling back to ONNX Runtime...")
                runner = ONNXInference(ONNX_PATH)
                runtime_name = "ONNX"
    else:
        print("[Runtime] Loading ONNX Runtime...")
        if HAS_ONNX:
            runner = ONNXInference(ONNX_PATH)
            runtime_name = "ONNX"
        else:
            print("ERROR: No runtime available")
            sys.exit(1)
    
    if not runner:
        print("ERROR: Could not initialize any runtime")
        sys.exit(1)
    
    print(f"  ✓ Using {runtime_name}")
    print()
    
    # Run requested test
    if args.image:
        test_single_image(runner, args.image)
    elif args.benchmark:
        benchmark_inference(runner, num_runs=args.runs)
    elif args.evaluate:
        print("[Evaluate] OCRBench evaluation not yet implemented")
        print("TODO: Load OCRBench test set and evaluate")
    else:
        # Default: quick test
        print("[Default] Running quick test with random input...")
        benchmark_inference(runner, num_runs=5)
    
    print()
    print("="*60)
    print("  ✓ Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()
