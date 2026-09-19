#!/bin/bash
# ===================================================================
# Area - One-command setup
# Run: chmod +x setup.sh && ./setup.sh
# ===================================================================
set -e

# Save the directory this script lives in (works even if called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "Area - Setup"
echo "==================================================="
echo ""

# -- Check Python --
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Python $PYTHON_VERSION found"

# -- Create venv if not exists --
cd "$SCRIPT_DIR"
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
echo "[OK] Virtual environment activated"

# -- Install Python dependencies --
echo ""
echo "[SETUP] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[OK] Python dependencies installed"

# -- Area-3R runtime check --
echo ""
AREA_3R_RUNTIME_DIR="$SCRIPT_DIR/area_3r_runtime"
if [ -d "$AREA_3R_RUNTIME_DIR" ]; then
    echo "[OK] Area-3R runtime found at $AREA_3R_RUNTIME_DIR"
else
    echo "[INFO] Area-3R runtime not found at $AREA_3R_RUNTIME_DIR"
    echo "[INFO] Place the Area-3R runtime there to enable geometric matching."
fi

# -- Verify bundled model weights (no internet download required) --
echo ""
echo "[SETUP] Verifying bundled model weights..."
if [ -f "$SCRIPT_DIR/Area-loc.safetensors" ]; then
    echo "[OK] Area-loc weights found"
else
    echo "[WARN] Area-loc.safetensors missing"
fi
if [ -f "$SCRIPT_DIR/Area-3R/model.safetensors" ]; then
    echo "[OK] Area-3R weights found"
else
    echo "[WARN] Area-3R/model.safetensors missing"
fi

# -- Create data directories --
echo ""
cd "$SCRIPT_DIR"
mkdir -p area_data/area_parts
mkdir -p area_data/index
echo "[OK] Data directories created"

# -- Done --
echo ""
echo "==================================================="
echo "  Setup complete!"
echo ""
echo "  To run Area:"
echo "    source venv/bin/activate"
echo "    python3 test_super.py"
echo "==================================================="