#!/bin/bash

# ============================================================================
# IDENTIFACE API - Installation Script
# Installs all dependencies and requirements
# Compatible with Python 3.10
# ============================================================================

set -e  # Exit on error

echo "=================================================="
echo "IDENTIFACE API v2.0 - Installation Script"
echo "=================================================="
echo ""

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "✗ Error: Python 3.10 is not installed"
    echo "Install it with: sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment with Python 3.10..."
    python3.10 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Verify Python version in venv
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Virtual environment Python: $VENV_PYTHON_VERSION"

# Upgrade pip, setuptools, wheel
echo ""
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Upgraded successfully"

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
echo "⏳ This may take 10-20 minutes (first time download)..."
echo ""

if pip install -r requirements.txt; then
    echo ""
    echo "✓ All dependencies installed successfully!"
else
    echo ""
    echo "✗ Installation failed. Check errors above."
    exit 1
fi

# Verify critical packages
echo ""
echo "Verifying critical packages..."
python -c "
import fastapi
import cv2
import numpy
import insightface
print('✓ FastAPI')
print('✓ OpenCV')
print('✓ NumPy')
print('✓ InsightFace')
" || {
    echo "✗ Verification failed"
    exit 1
}

echo ""
echo "=================================================="
echo "✓ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the API server:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "2. Test the API (in another terminal):"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Access documentation:"
echo "   Swagger UI: http://localhost:8000/docs"
echo "   ReDoc: http://localhost:8000/redoc"
echo ""
echo "For detailed setup instructions, see SETUP_GUIDE.md"
echo ""
