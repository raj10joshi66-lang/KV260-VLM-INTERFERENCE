set -e

# ─── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Configuration ───────────────────────────────────────────
DOCKER_IMAGE="xilinx/vitis-ai-pytorch:latest"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FP32="vision_encoder.onnx"
SCRIPT_NAME="compile_model.sh"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Phase 4: KV260 DPU Model Compilation${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}\n"

# STEP 1: Pre-Flight Checks

echo -e "${YELLOW}[Step 1/5] Pre-flight checks${NC}"

# Check FP32 ONNX model (for reference - will use INT8 if available)
if [ ! -f "$WORKSPACE_DIR/$MODEL_FP32" ]; then
    echo -e "${RED}  ✗ FP32 ONNX not found: $MODEL_FP32${NC}"
    echo "    Expected: $(du -h $MODEL_FP32 2>/dev/null || echo 'not found')"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} FP32 ONNX: $(du -h ${WORKSPACE_DIR}/${MODEL_FP32} | cut -f1)"

# Check compilation script
if [ ! -f "$WORKSPACE_DIR/$SCRIPT_NAME" ]; then
    echo -e "${RED}  ✗ Compilation script not found: $SCRIPT_NAME${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Compilation script exists"

# Check for INT8 model (preferred)
if [ -f "$WORKSPACE_DIR/vision_encoder_int8.onnx" ]; then
    MODEL_TO_COMPILE="vision_encoder_int8.onnx"
    echo -e "  ${GREEN}✓${NC} INT8 ONNX found: $(du -h ${WORKSPACE_DIR}/vision_encoder_int8.onnx | cut -f1)"
elif [ -f "$WORKSPACE_DIR/vision_encoder_int8_simple.onnx" ]; then
    MODEL_TO_COMPILE="vision_encoder_int8_simple.onnx"
    echo -e "  ${YELLOW}⚠${NC} Using test INT8: $(du -h ${WORKSPACE_DIR}/vision_encoder_int8_simple.onnx | cut -f1)"
else
    echo -e "${RED}  ✗ No INT8 model found${NC}"
    echo "    Options:"
    echo "      1. Run Phase 3: python quantize_model.py"
    echo "      2. Use FP32 with Docker quantization"
    echo ""
    read -p "    Continue with FP32? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    MODEL_TO_COMPILE="$MODEL_FP32"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}  ✗ Docker not found${NC}"
    echo "    Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Docker available: $(docker --version)"

echo ""

# ═════════════════════════════════════════════════════════════
# STEP 2: Pull Vitis-AI Docker Image
# ═════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Step 2/5] Checking Vitis-AI Docker image${NC}"

if docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Image already pulled"
    IMAGE_SIZE=$(docker image inspect "$DOCKER_IMAGE" | grep -o '"Size": [0-9]*' | head -1 | cut -d' ' -f2)
    echo "    Size: $((IMAGE_SIZE / 1024 / 1024 / 1024)) GB"
else
    echo -e "  ${YELLOW}Pulling Docker image...${NC}"
    echo "    (This may take 5-10 minutes on first run)"
    docker pull "$DOCKER_IMAGE"
    echo -e "  ${GREEN}✓${NC} Image pulled successfully"
fi

echo ""

# ═════════════════════════════════════════════════════════════
# STEP 3: Display Docker Command
# ═════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Step 3/5] Docker command to run${NC}"
echo ""
echo -e "${BLUE}Copy and paste this command to start the container:${NC}"
echo ""
echo -e "${GREEN}docker run -it --rm \\${NC}"
echo -e "${GREEN}  -v $WORKSPACE_DIR:/workspace \\${NC}"
echo -e "${GREEN}  -w /workspace \\${NC}"
echo -e "${GREEN}  $DOCKER_IMAGE${NC}"
echo ""
echo -e "${BLUE}Then inside the container, run:${NC}"
echo ""
echo -e "${GREEN}conda activate vitis-ai-pytorch${NC}"
echo -e "${GREEN}chmod +x compile_model.sh${NC}"
echo -e "${GREEN}./compile_model.sh${NC}"
echo ""

# ═════════════════════════════════════════════════════════════
# STEP 4: Interactive Mode Option
# ═════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Step 4/5] Options${NC}"
echo ""
echo "  1. ${BLUE}Copy & Paste${NC}"
echo "     Copy the Docker command above and run manually"
echo ""
echo "  2. ${BLUE}Automatic (Recommended)${NC}"
echo "     Let this script start the container for you"
echo ""
echo "  3. ${BLUE}Skip${NC}"
echo "     You'll run Docker manually"
echo ""

read -p "Choose option (1-3): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo -e "${YELLOW}Please copy and run this Docker command:${NC}"
        echo ""
        echo "docker run -it --rm -v $WORKSPACE_DIR:/workspace -w /workspace $DOCKER_IMAGE"
        echo ""
        exit 0
        ;;
    2)
        echo ""
        echo -e "${YELLOW}[Step 5/5] Starting Docker container...${NC}"
        echo ""
        
        docker run -it --rm \
            -v "$WORKSPACE_DIR:/workspace" \
            -w /workspace \
            "$DOCKER_IMAGE" \
            bash -c "
                echo ''
                echo '═══════════════════════════════════════════════════════'
                echo '  Inside Vitis-AI Docker Container'
                echo '═══════════════════════════════════════════════════════'
                echo ''
                
                # Activate environment
                source /opt/vitis_ai/conda/etc/profile.d/conda.sh
                conda activate vitis-ai-pytorch
                
                echo '[1/3] Vitis-AI environment activated'
                echo ''
                
                # Give permission
                chmod +x compile_model.sh
                echo '[2/3] compile_model.sh made executable'
                echo ''
                
                # Run compilation
                echo '[3/3] Running compilation...'
                echo ''
                ./compile_model.sh
                
                echo ''
                echo '═══════════════════════════════════════════════════════'
                echo '  Compilation Pipeline Complete'
                echo '═══════════════════════════════════════════════════════'
                echo ''
                echo 'Output files:'
                ls -lh compiled_model/ 2>/dev/null || echo '  (Check compilation output above)'
                echo ''
            "
        
        # Check results
        echo ""
        if [ -f "$WORKSPACE_DIR/compiled_model/vision_encoder.xmodel" ]; then
            echo -e "${GREEN}✅ Compilation successful!${NC}"
            echo -e "   Output: $(du -h $WORKSPACE_DIR/compiled_model/vision_encoder.xmodel | cut -f1)"
        else
            echo -e "${YELLOW}⚠ Check compilation output above for errors${NC}"
        fi
        echo ""
        exit 0
        ;;
    3)
        echo ""
        echo -e "${YELLOW}Manual Docker execution:${NC}"
        echo ""
        echo "docker run -it --rm -v $WORKSPACE_DIR:/workspace -w /workspace $DOCKER_IMAGE"
        echo ""
        echo "Then inside container:"
        echo "  conda activate vitis-ai-pytorch"
        echo "  ./compile_model.sh"
        echo ""
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
