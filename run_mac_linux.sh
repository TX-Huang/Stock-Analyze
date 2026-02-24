#!/bin/bash
echo "=========================================="
echo "     Alpha Global Quantitative Platform"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "Error: Python3 could not be found. Please install Python3."
    exit 1
fi

PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $PYTHON_VER"

if [[ "$PYTHON_VER" != "3.11"* ]]; then
    echo ""
    echo "[WARNING] ---------------------------------------------------"
    echo "It is highly recommended to use Python 3.11 for Finlab stability."
    echo "Your current version is $PYTHON_VER."
    echo "If you encounter errors, please use Python 3.11."
    echo "-------------------------------------------------------------"
    echo ""
    read -p "Press Enter to continue..."
fi

echo "[1/3] Creating virtual environment (optional but recommended)..."
# Check if venv exists, if not create
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

echo "[2/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[3/3] Launching App..."
echo "The App will open in your default browser shortly."
echo "Press Ctrl+C to stop."
echo ""

streamlit run app.py
