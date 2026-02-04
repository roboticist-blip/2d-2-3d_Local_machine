#!/bin/bash
# Installation script for 3D Gaussian Splatting Local Toolkit

set -e

echo "=========================================="
echo "Installing 3D Gaussian Splatting Toolkit"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "⚠️  No NVIDIA GPU detected. Training will not work without GPU."
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Python dependencies installed"

# Install COLMAP
echo ""
echo "Checking for COLMAP..."
if ! command -v colmap &> /dev/null; then
    echo "Installing COLMAP..."
    
    # Try apt-get first (Ubuntu/Debian)
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y colmap
    else
        echo "⚠️  Please install COLMAP manually:"
        echo "   Ubuntu/Debian: sudo apt-get install colmap"
        echo "   macOS: brew install colmap"
        echo "   Or build from source: https://colmap.github.io/install.html"
        exit 1
    fi
fi

echo "✓ COLMAP installed: $(colmap -h 2>&1 | head -1)"

# Install FFmpeg
echo ""
echo "Checking for FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y ffmpeg
    else
        echo "⚠️  Please install FFmpeg manually:"
        echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "   macOS: brew install ffmpeg"
        exit 1
    fi
fi

echo "✓ FFmpeg installed: $(ffmpeg -version | head -1)"

# Clone Gaussian Splatting repository
echo ""
echo "Setting up 3D Gaussian Splatting..."
GS_PATH="$HOME/gaussian-splatting"

if [ -d "$GS_PATH" ]; then
    echo "✓ Gaussian Splatting already exists at $GS_PATH"
else
    echo "Cloning Gaussian Splatting repository..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive "$GS_PATH"
    
    cd "$GS_PATH"
    
    # Install submodules
    echo "Installing Gaussian Splatting submodules..."
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
    
    cd -
    
    echo "✓ Gaussian Splatting installed at $GS_PATH"
fi

# Install CLI tools
echo ""
echo "Installing CLI commands..."

# Get the absolute path to this script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make CLI scripts executable
chmod +x "$SCRIPT_DIR/gs_toolkit/cli/gs_process.py"
chmod +x "$SCRIPT_DIR/gs_toolkit/cli/gs_train.py"
chmod +x "$SCRIPT_DIR/gs_toolkit/cli/gs_export.py"

# Create symlinks in /usr/local/bin
echo "Creating command symlinks (requires sudo)..."
sudo ln -sf "$SCRIPT_DIR/gs_toolkit/cli/gs_process.py" /usr/local/bin/gs-process
sudo ln -sf "$SCRIPT_DIR/gs_toolkit/cli/gs_train.py" /usr/local/bin/gs-train
sudo ln -sf "$SCRIPT_DIR/gs_toolkit/cli/gs_export.py" /usr/local/bin/gs-export

echo "✓ CLI commands installed"

# Test installation
echo ""
echo "=========================================="
echo "Testing Installation"
echo "=========================================="

python3 -c "import torch; print('✓ PyTorch:', torch.__version__)"
python3 -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
python3 -c "import plyfile; print('✓ PLYfile installed')"
colmap -h > /dev/null 2>&1 && echo "✓ COLMAP works" || echo "✗ COLMAP failed"
ffmpeg -version > /dev/null 2>&1 && echo "✓ FFmpeg works" || echo "✗ FFmpeg failed"

echo ""
gs-process --help > /dev/null 2>&1 && echo "✓ gs-process works" || echo "✗ gs-process failed"
gs-train --help > /dev/null 2>&1 && echo "✓ gs-train works" || echo "✗ gs-train failed"
gs-export --help > /dev/null 2>&1 && echo "✓ gs-export works" || echo "✗ gs-export failed"

echo ""
echo "=========================================="
echo "Installation Complete! ✓"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  1. gs-process video --data video.mp4 --out ./data"
echo "  2. gs-train ./data"
echo "  3. gs-export ./data/output --ply --video"
echo ""
echo "See README.md for complete documentation"
