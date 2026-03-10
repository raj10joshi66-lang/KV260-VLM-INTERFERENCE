#!/bin/bash
# ============================================================
#  Phase 4: Compile quantized ONNX model for KV260 DPU
#  Run INSIDE the Vitis-AI Docker container:
#    chmod +x compile_model.sh && ./compile_model.sh
# ============================================================

set -e  # exit on any error

# ─── Paths ───────────────────────────────────────────────────
WORKSPACE_DIR="/workspace"
MODEL_IN="${WORKSPACE_DIR}/vision_encoder_int8.onnx"
OUT_DIR="${WORKSPACE_DIR}"
MODEL_NAME="vision_encoder"

# ─── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Phase 4: KV260 DPU Model Compilation${NC}"
echo -e "${GREEN}=====================================================${NC}"

# ─── Pre-flight checks ───────────────────────────────────────
echo -e "\n${YELLOW}[Pre-flight checks]${NC}"

if [ ! -f "${MODEL_IN}" ]; then
    echo -e "${RED}ERROR: Quantized model not found: ${MODEL_IN}${NC}"
    echo "Expected: vision_encoder_int8.onnx from Phase 3"
    exit 1
fi
echo "  ✓ INT8 ONNX model: $(du -h ${MODEL_IN} | cut -f1)"

# Verify Vitis-AI environment
if ! command -v vai_c_xir &> /dev/null; then
    echo -e "${RED}ERROR: vai_c_xir not found${NC}"
    echo "Make sure you're inside Vitis-AI Docker with conda activated"
    echo "  conda activate vitis-ai-pytorch"
    exit 1
fi
echo "  ✓ Vitis-AI tools available"

# Find KV260 architecture JSON
ARCH_JSON=""
if [ -f "/opt/vitis_ai/compiler_board_package/kv260/arch.json" ]; then
    ARCH_JSON="/opt/vitis_ai/compiler_board_package/kv260/arch.json"
elif [ -f "/vitis_ai_home/compiler_board_package/kv260/arch.json" ]; then
    ARCH_JSON="/vitis_ai_home/compiler_board_package/kv260/arch.json"
else
    # Search for it
    ARCH_JSON=$(find / -name "kv260*arch.json" -o -name "KV260*arch.json" 2>/dev/null | head -1)
fi

if [ -z "${ARCH_JSON}" ] || [ ! -f "${ARCH_JSON}" ]; then
    echo -e "${RED}ERROR: KV260 arch.json not found${NC}"
    echo "Standard locations:"
    echo "  /opt/vitis_ai/compiler_board_package/kv260/arch.json"
    echo "  /vitis_ai_home/compiler_board_package/kv260/arch.json"
    exit 1
fi
echo "  ✓ KV260 arch.json: $(basename ${ARCH_JSON})"

mkdir -p "${OUT_DIR}/compiled_model"
echo "  ✓ Output directory: ${OUT_DIR}/compiled_model"

# ─── MAIN: Compile to DPU-compatible format ─────────────────
echo -e "\n${YELLOW}[Main] Compiling INT8 ONNX for KV260...${NC}"
echo "  Input:  ${MODEL_IN}"
echo "  Output: ${OUT_DIR}/compiled_model/${MODEL_NAME}.xmodel"
echo ""

# Use vai_c_xir for direct ONNX → xmodel compilation
vai_c_xir \
    --xmodel "${MODEL_IN}" \
    --arch "${ARCH_JSON}" \
    --output_dir "${OUT_DIR}/compiled_model" \
    --net_name "${MODEL_NAME}" \
    2>&1 | tee "${OUT_DIR}/compilation.log"

# ─── Verify compilation ──────────────────────────────────────
echo -e "\n${YELLOW}[Verification]${NC}"

XMODEL="${OUT_DIR}/compiled_model/${MODEL_NAME}.xmodel"
if [ ! -f "${XMODEL}" ]; then
    echo -e "${RED}ERROR: Compilation failed — .xmodel not found${NC}"
    echo "Check compilation.log for details:"
    tail -50 "${OUT_DIR}/compilation.log"
    exit 1
fi

XMODEL_SIZE=$(du -h "${XMODEL}" | cut -f1)
echo "  ✓ Compiled model: ${XMODEL_SIZE}"

# Extract subgraph info
echo ""
echo -e "${YELLOW}[Subgraph Info]${NC}"
python3 << 'PYEOF'
import sys
try:
    import xir
    xmodel_path = "/workspace/compiled_model/vision_encoder.xmodel"
    g = xir.Graph.deserialize(xmodel_path)
    root = g.get_root_subgraph()
    subgraphs = list(root.toposort_child_subgraph())
    
    print(f"  Total subgraphs: {len(subgraphs)}")
    for i, sg in enumerate(subgraphs, 1):
        device = sg.get_attr('device') if sg.has_attr('device') else 'CPU'
        ops = len(list(sg.toposort_op()))
        print(f"    [{i}] {sg.get_name():20s} ({device}, {ops} ops)")
except Exception as e:
    print(f"  (Subgraph info unavailable: {e})")
PYEOF

# ─── Summary ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  ✅ Compilation Successful!${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo "Output:"
echo "  Location: ${OUT_DIR}/compiled_model/"
echo "  File: ${MODEL_NAME}.xmodel"
echo "  Size: ${XMODEL_SIZE}"
echo ""
echo "Next: Deploy to KV260 board"
echo "  1. Transfer files: bash transfer_to_kv260.sh <board_ip>"
echo "  2. Connect via SSH: ssh ubuntu@<board_ip>"
echo "  3. Run deployment: python3 deploy.py"
echo -e "${GREEN}=====================================================${NC}"

# ─── Save metadata ───────────────────────────────────────────
cat > "${OUT_DIR}/COMPILATION_INFO.txt" << EOF
KV260 DPU Compilation Report
=============================
Date: $(date)
Input Model: $(basename ${MODEL_IN})
Input Size: $(du -h ${MODEL_IN} | cut -f1)
Output Model: ${MODEL_NAME}.xmodel
Output Size: ${XMODEL_SIZE}
Architecture: KV260
Vitis-AI Version: $(vai_c_xir --version 2>/dev/null || echo "unknown")

Subgraph Mapping:
$(python3 -c "
import xir
try:
    g = xir.Graph.deserialize('${XMODEL}')
    for sg in g.get_root_subgraph().toposort_child_subgraph():
        device = sg.get_attr('device') if sg.has_attr('device') else 'CPU'
        print(f'  {sg.get_name()}: {device}')
except: pass
")

Status: ✓ Ready for deployment
EOF
echo "  Metadata saved: COMPILATION_INFO.txt"
