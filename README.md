# 3D Gaussian Splatting - Local Training Toolkit

**End-to-end pipeline: Clone → Install → Process → Train → Export with Metrics**

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/roboticist-blip/2d-2-3d_Local_machine.git
cd gs_local

# 2. (Recommended) Create & activate Conda environment
# Creates an environment named `2d-2-3d` with Python 3.10
conda create -n 2d-2-3d python=3.10 pip -y
conda activate 2d-2-3d

# 3. Install requirements
chmod +x install.sh
./install.sh

# 4. Process data
gs-process video --data {path/to/video.mp4} --out {path/to/output}

# 5. Train model
gs-train ./data/scene1

# 6. Export with evaluation metrics
gs-export ./data/scene1/output --ply --video
```

## 📋 Complete Workflow

### Step 1: Clone Repository
```bash
git clone https://github.com/your-repo/gs_local.git
cd gs_local
```

### Step 2: Install Requirements
```bash
# Recommended: create and activate the project's Conda environment first
# (creates `2d-2-3d` if it doesn't exist)
conda create -n 2d-2-3d python=3.10 pip -y
conda activate 2d-2-3d

chmod +x install.sh
./install.sh
```

Installs: PyTorch, COLMAP (when available via package manager), FFmpeg, Gaussian Splatting, CLI tools

### Step 3: Process Data
```bash
gs-process video --data video.mp4 --out ./data/scene --lightweight
```

Output: Frames + COLMAP camera poses

### Step 4: Train Model
```bash
gs-train ./data/scene --lightweight
```

Trains for 30k iterations (~1-2 hours on RTX 3090)

### Step 5: Export with Metrics
```bash
gs-export ./data/scene/output --ply --video
```

Outputs:
- PLY file
- Rendered video
- **Evaluation metrics** (PSNR, SSIM, LPIPS)

## 📊 Evaluation Metrics

Automatically computed:

**Rendering Quality:**
- PSNR: 25-35 dB (higher is better)
- SSIM: 0.85-0.95 (closer to 1 is better)  
- LPIPS: 0.05-0.15 (lower is better)

**Model Statistics:**
- Number of Gaussians
- File size
- Scene extent
- Opacity distribution

Saved to: `evaluation_metrics.json`

## 🎬 Camera Paths

```bash
# Orbit
gs-export ./output --video --camera-path orbit --orbit-elevation 30

# Spiral
gs-export ./output --video --camera-path spiral --spiral-loops 2

# Linear dolly
gs-export ./output --video --camera-path linear \
    --linear-start 0 0 5 --linear-end 5 2 0
```

## 💡 Lightweight Mode

40% smaller models, 25% faster:
```bash
gs-process video --data video.mp4 --out ./data --lightweight
gs-train ./data --lightweight
```

## 📁 Project Structure

```
gs_local/
├── README.md
├── install.sh              # Installation
├── requirements.txt        # Dependencies
├── gs_toolkit/
│   ├── cli/               # Commands
│   ├── core/              # Core modules
│   └── utils/             # Utilities
└── data/                  # Your data
    └── scene/
        ├── images/        # Frames
        ├── sparse/        # COLMAP
        ├── output/        # Model
        └── exports/       # PLY + metrics
```

## 📖 Full Documentation

See complete guide: [QUICKREF.md](QUICKREF.md)

## 🐛 Troubleshooting

**COLMAP fails:** Use more frames `--max-frames 500`
**Out of memory:** Use `--lightweight` on both commands
**Low quality:** Train longer `--iterations 50000`

## 🎓 For Research

All metrics saved in JSON for publication:
- PSNR, SSIM, LPIPS
- Model statistics
- Training parameters

**Citation:**
```bibtex
@article{kerbl20233d,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and others},
  journal={ACM TOG},
  year={2023}
}
```

## ⚡ Requirements

- NVIDIA GPU (8GB+ VRAM)
- Python 3.8+
- Ubuntu/Debian/macOS
