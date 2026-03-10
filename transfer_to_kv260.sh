#!/bin/bash
# ============================================================
#  Phase 5: Transfer compiled model to KV260 board
#  Usage: bash transfer_to_kv260.sh <board_ip>
#  Example: bash transfer_to_kv260.sh 192.168.1.100
# ============================================================

set -e

BOARD_IP="${1:-}"
BOARD_USER="ubuntu"
BOARD_DIR="/home/ubuntu/kv260-vlm"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Validate input ───────────────────────────────────────────
if [ -z "${BOARD_IP}" ]; then
    echo -e "${YELLOW}KV260 Board File Transfer${NC}"
    echo ""
    echo "Usage: bash transfer_to_kv260.sh <board_ip>"
    echo ""
    echo "Examples:"
    echo "  bash transfer_to_kv260.sh 192.168.1.100"
    echo "  bash transfer_to_kv260.sh kv260-board"
    echo ""
    echo "Setup SSH (optional, for easier access):"
    echo "  Edit ~/.ssh/config:"
    echo ""
    echo "  Host kv260-board"
    echo "    HostName 192.168.1.100"
    echo "    User ubuntu"
    echo "    Port 22"
    echo ""
    echo "  Then use: bash transfer_to_kv260.sh kv260-board"
    exit 0
fi

echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Phase 5: Transfer to KV260 Board${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo "Board: ${BOARD_USER}@${BOARD_IP}"
echo "Local: ${LOCAL_DIR}"
echo "Remote: ${BOARD_DIR}"
echo ""

# ─── Verify connectivity ─────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking board connectivity...${NC}"
if ! ssh -o ConnectTimeout=5 "${BOARD_USER}@${BOARD_IP}" "echo OK" &>/dev/null; then
    echo -e "${RED}ERROR: Cannot connect to board${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check board IP: ${BOARD_IP}"
    echo "  2. Ensure board is powered on and on the same network"
    echo "  3. Test SSH: ssh ubuntu@${BOARD_IP}"
    echo "  4. Default password: xilinx"
    exit 1
fi
echo "  ✓ Board reachable at ${BOARD_IP}"

# ─── Verify local files ──────────────────────────────────────
echo -e "\n${YELLOW}[2/6] Checking local files...${NC}"

HAS_XMODEL=false
HAS_INT8=false
HAS_PT=false

if [ -f "${LOCAL_DIR}/compiled_model/vision_encoder.xmodel" ]; then
    echo "  ✓ Compiled xmodel: $(du -h ${LOCAL_DIR}/compiled_model/vision_encoder.xmodel | cut -f1)"
    HAS_XMODEL=true
else
    echo "  ✗ No xmodel found (expected from Phase 4)"
fi

if [ -f "${LOCAL_DIR}/vision_encoder_int8.onnx" ]; then
    echo "  ✓ INT8 ONNX: $(du -h ${LOCAL_DIR}/vision_encoder_int8.onnx | cut -f1)"
    HAS_INT8=true
fi

if [ -d "${LOCAL_DIR}/smolvlm2_model" ]; then
    SIZE=$(du -sh ${LOCAL_DIR}/smolvlm2_model | cut -f1)
    echo "  ✓ Model weights: ${SIZE}"
    HAS_PT=true
fi

if [ "$HAS_XMODEL" = false ] && [ "$HAS_INT8" = false ]; then
    echo -e "${RED}ERROR: No compiled model found${NC}"
    echo "Run Phase 3 (quantization) and Phase 4 (compilation) first"
    exit 1
fi

# ─── Create remote directories ───────────────────────────────
echo -e "\n${YELLOW}[3/6] Creating remote directories...${NC}"
ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p ${BOARD_DIR}/{compiled_model,models,deployment}"
echo "  ✓ Created directory structure"

# ─── Transfer compiled xmodel ────────────────────────────────
if [ "$HAS_XMODEL" = true ]; then
    echo -e "\n${YELLOW}[4/6] Transferring compiled xmodel...${NC}"
    scp "${LOCAL_DIR}/compiled_model/vision_encoder.xmodel" \
        "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/compiled_model/"
    echo "  ✓ xmodel transferred"
fi

# ─── Transfer INT8 ONNX (fallback/evaluation) ────────────────
if [ "$HAS_INT8" = true ]; then
    echo -e "\n${YELLOW}[5/6] Transferring INT8 ONNX model...${NC}"
    scp "${LOCAL_DIR}/vision_encoder_int8.onnx" \
        "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/"
    echo "  ✓ INT8 ONNX transferred"
fi

# ─── Transfer model weights (if available) ──────────────────
if [ "$HAS_PT" = true ]; then
    READ_SIZE=$(du -sh ${LOCAL_DIR}/smolvlm2_model | cut -f1)
    echo -e "\n${YELLOW}[6/6] Transferring model weights (${READ_SIZE})...${NC}"
    echo "  (This may take several minutes...)"
    scp -r "${LOCAL_DIR}/smolvlm2_model" \
        "${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/"
    echo "  ✓ Model weights transferred"
else
    echo -e "\n${YELLOW}[6/6] Model weights not found locally${NC}"
    echo "  → Board will download from HuggingFace on first run"
fi

# ─── Create remote setup script ──────────────────────────────
echo -e "\n${YELLOW}[Extra] Installing dependencies on board...${NC}"

ssh "${BOARD_USER}@${BOARD_IP}" << 'SSH_SCRIPT'
set -e
echo "Installing Python dependencies..."
pip3 install transformers==4.40.0 pillow numpy onnxruntime --quiet 2>/dev/null || echo "Some packages may be pre-installed"
echo "✓ Dependencies ready"
SSH_SCRIPT

# ─── Verify remote files ─────────────────────────────────────
echo -e "\n${YELLOW}[Verification] Checking remote files...${NC}"
ssh "${BOARD_USER}@${BOARD_IP}" "ls -lh ${BOARD_DIR}/" | head -10

# ─── Summary ─────────────────────────────────────────────────
echo -e "\n${GREEN}=====================================================${NC}"
echo -e "${GREEN}  ✅ Transfer Complete!${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo "Remote location: ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Connect to board:"
echo "     ${BLUE}ssh ${BOARD_USER}@${BOARD_IP}${NC}"
echo ""
echo "  2. Navigate to project:"
echo "     ${BLUE}cd ${BOARD_DIR}${NC}"
echo ""
echo "  3. Create deployment script:"
echo "     ${BLUE}cat > deploy.py << 'EOF'${NC}"
echo "     ..."
echo "     ${BLUE}EOF${NC}"
echo ""
echo "  4. Run inference:"
echo "     ${BLUE}python3 deploy.py${NC}"
echo ""
echo "Alternative: VS Code Remote"
echo "  - Install Remote - SSH extension"
echo "  - Add: Host kv260-board"
echo "  - Connect to ${BOARD_IP}"
echo "  - Edit files directly on board"
echo ""
echo -e "${GREEN}=====================================================${NC}"
