set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "═══════════════════════════════════════════════════════════════"
echo "  KV260 SmolVLM2 - Phase 3: INT8 Quantization (Vitis-AI)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Check prerequisites
echo "[1/5] Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker available: $(docker --version)"

if [ ! -f "$PROJECT_DIR/vision_encoder.onnx" ]; then
    echo "❌ vision_encoder.onnx not found"
    echo "   Run Phases 1-2 first: python download_model.py && python split_model.py && python export_onnx.py"
    exit 1
fi
echo "✓ vision_encoder.onnx found (1.6M)"

if [ ! -d "$PROJECT_DIR/calibration_dataset" ] || [ $(ls -1 "$PROJECT_DIR/calibration_dataset"/*.jpg 2>/dev/null | wc -l) -lt 250 ]; then
    echo "❌ Calibration dataset incomplete"
    echo "   Run: python prepare_calibration_data.py"
    exit 1
fi
echo "✓ Calibration dataset ready (250 images)"

# Pull Docker image
echo ""
echo "[2/5] Pulling Vitis-AI Docker image..."
echo "   (This may take 2-5 minutes on first run...)"
docker pull xilinx/vitis-ai-pytorch:latest > /dev/null
echo "✓ Vitis-AI image available"

# Run quantization
echo ""
echo "[3/5] Starting Vitis-AI container..."
echo "   Location: $PROJECT_DIR"
echo ""

docker run -it --rm \
    -v "$PROJECT_DIR:/workspace" \
    xilinx/vitis-ai-pytorch:latest \
    bash -c "
        cd /workspace
        
        echo '══════════════════════════════════════════════'
        echo '  Inside Vitis-AI Container'
        echo '══════════════════════════════════════════════'
        echo ''
        
        # Verify environment
        echo '[4/5] Setting up Vitis-AI environment...'
        conda activate vitis-ai-pytorch
        python -c 'import vai_q_onnx; print(\"✓ vai_q_onnx available\")'
        
        # Run quantization
        echo ''
        echo '[5/5] Running INT8 quantization...'
        python quantize_model.py
    "

# Verify output
echo ""
echo "[Final] Verifying quantized model..."
if [ -f "$PROJECT_DIR/vision_encoder_int8.onnx" ]; then
    SIZE=$(du -h "$PROJECT_DIR/vision_encoder_int8.onnx" | cut -f1)
    echo "✅ Quantization complete!"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  RESULT"
    echo "═══════════════════════════════════════════════════════════════"
    ls -lh "$PROJECT_DIR"/vision_encoder*.onnx | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "Next step: Xilinx DPU Compiler for KV260 deployment"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "⚠️  Quantization completed but vision_encoder_int8.onnx not found"
    echo "    Check Docker output above for errors"
    exit 1
fi
