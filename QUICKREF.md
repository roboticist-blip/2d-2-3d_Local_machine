# Quick Reference Card

## Three Commands You Need

```bash
# 1. PROCESS DATA
gs-process {video|images} --data INPUT --out OUTPUT

# 2. TRAIN MODEL  
gs-train DATA_PATH

# 3. EXPORT RESULTS
gs-export MODEL_PATH --ply --video --camera-path orbit
```

## Common Workflows

### Basic (Video → PLY)
```bash
gs-process video --data video.mp4 --out ./data
gs-train ./data
gs-export ./data/output --ply
```

### With Custom Camera Video
```bash
gs-process video --data video.mp4 --out ./data --lightweight
gs-train ./data --lightweight
gs-export ./data/output --ply --video --camera-path spiral
```

### From Images
```bash
gs-process images --data ./photos --out ./data
gs-train ./data
gs-export ./data/output --ply --video
```

## Key Options

### gs-process
```
--fps 2                    # Frames per second
--max-frames 300           # Max frames to extract
--resolution 1920 1080     # Target resolution
--lightweight              # Smaller, faster
--quality 95               # JPEG quality
```

### gs-train
```
--iterations 30000         # Training iterations (7k/15k/30k/50k)
--lightweight              # 40% smaller model
--gpu 0                    # GPU device
```

### gs-export
```
--ply                      # Export PLY file
--video                    # Render video
--camera-path orbit        # orbit|spiral|linear|custom
--orbit-elevation 30       # Camera angle
--num-frames 240           # Video frames
--fps 30                   # Video framerate
--resolution 1920 1080     # Video resolution
```

## Camera Paths

### Orbit (Default)
```bash
gs-export ./output --video --camera-path orbit --orbit-elevation 20
```

### Spiral  
```bash
gs-export ./output --video --camera-path spiral --spiral-loops 2
```

### Linear Dolly
```bash
gs-export ./output --video --camera-path linear \
    --linear-start 0 0 5 --linear-end 5 2 0
```

### Custom JSON
```bash
# Create camera.json with custom positions
gs-export ./output --video --camera-path custom --camera-config camera.json
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| COLMAP fails | More frames: `--max-frames 500` |
| Out of memory | Lightweight: `--lightweight` on both commands |
| Blurry output | More iterations: `--iterations 50000` |
| Slow rendering | Lower res: `--resolution 1280 720` |

## File Locations

```
OUTPUT_DIR/
├── images/              # Extracted frames
├── sparse/              # COLMAP camera poses  
├── output/              # Trained model
│   └── point_cloud/
│       └── iteration_30000/
│           └── point_cloud.ply
└── exports/
    ├── model_iter_30000.ply
    └── render_iter_30000.mp4
```

## Installation

```bash
git clone <repo> gs_local
cd gs_local
chmod +x setup.sh
./setup.sh
```

## Performance

| GPU | Time (30k iter) | Model Size |
|-----|-----------------|------------|
| RTX 4090 | ~45 min | 150-400 MB |
| RTX 3090 | ~1.5 hr | 150-400 MB |
| RTX 3080 | ~2 hr | 150-400 MB |

*Add `--lightweight` to reduce size ~40% and time ~25%*

## Tips

1. **Video capture**: Circle 360° around object, smooth motion
2. **Lighting**: Diffuse, avoid harsh shadows
3. **Duration**: 30-60 seconds
4. **Preview**: Use `--iterations 7000` for quick test
5. **Quality**: 30k iterations for publication
6. **Storage**: Lightweight mode for 40% smaller files

## Help

```bash
gs-process --help
gs-train --help  
gs-export --help
```

See README.md for full documentation.
