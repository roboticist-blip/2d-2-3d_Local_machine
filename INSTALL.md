# Installation & Complete Code Guide

This guide contains ALL the code you need. Simply copy each section to the specified file.

## Step 1: Create Project Structure

```bash
mkdir -p gs_local/gs_toolkit/{cli,core,utils}
mkdir -p gs_local/{docs,examples}
cd gs_local
```

## Step 2: Copy Core Modules

### File: `gs_toolkit/core/video_processor.py`

```python
"""Video processing module for frame extraction"""
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional
import logging

logger = logging.getLogger('gs-process')

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, lightweight: bool = False):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.lightweight = lightweight
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {'video_path': str(self.video_path), 'lightweight': lightweight, 'frames': []}
    
    def extract_frames(self, fps: Optional[float] = None, max_frames: Optional[int] = None,
                      resolution: Optional[Tuple[int, int]] = None, quality: int = 95) -> int:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps is None:
            fps = orig_fps
        
        frame_interval = max(1, int(orig_fps / fps))
        if max_frames:
            frame_interval = max(frame_interval, total_frames // max_frames)
        
        if self.lightweight and quality > 90:
            quality = 90
        
        frame_count = extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                if resolution:
                    frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LANCZOS4)
                
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(str(self.output_dir / frame_filename), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                self.metadata['frames'].append({
                    'filename': frame_filename,
                    'frame_index': frame_count,
                    'timestamp': frame_count / orig_fps
                })
                
                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return extracted_count
```

### File: `gs_toolkit/core/colmap_processor.py`

```python
"""COLMAP processor"""
import subprocess
import shutil
from pathlib import Path
import logging

logger = logging.getLogger('gs-process')

class ColmapProcessor:
    def __init__(self, images_dir: str, output_dir: str, quality: str = 'high',
                 use_gpu: bool = True, single_camera: bool = True, lightweight: bool = False):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.use_gpu = use_gpu
        self.single_camera = single_camera
        self.lightweight = lightweight
        
        self.database_path = self.output_dir / 'database.db'
        self.sparse_dir = self.output_dir / 'sparse'
        self.sparse_dir.mkdir(exist_ok=True)
        
        if not shutil.which('colmap'):
            raise RuntimeError("COLMAP not found")
    
    def process(self) -> bool:
        try:
            if not self._feature_extraction():
                return False
            if not self._feature_matching():
                return False
            if not self._mapper():
                return False
            return True
        except Exception as e:
            logger.error(f"COLMAP failed: {e}")
            return False
    
    def _feature_extraction(self) -> bool:
        quality_presets = {
            'low': {'max_image_size': 1600, 'max_num_features': 4096},
            'medium': {'max_image_size': 2400, 'max_num_features': 8192},
            'high': {'max_image_size': 3200, 'max_num_features': 16384},
            'extreme': {'max_image_size': 4800, 'max_num_features': 32768}
        }
        
        if self.lightweight and self.quality in ['high', 'extreme']:
            self.quality = 'medium'
        
        preset = quality_presets.get(self.quality, quality_presets['high'])
        
        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.single_camera', '1' if self.single_camera else '0',
            '--SiftExtraction.use_gpu', '1' if self.use_gpu else '0',
            '--SiftExtraction.max_image_size', str(preset['max_image_size']),
            '--SiftExtraction.max_num_features', str(preset['max_num_features'])
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def _feature_matching(self) -> bool:
        matcher = 'sequential' if self.lightweight else 'exhaustive'
        
        if matcher == 'exhaustive':
            cmd = ['colmap', 'exhaustive_matcher',
                  '--database_path', str(self.database_path),
                  '--SiftMatching.use_gpu', '1' if self.use_gpu else '0']
        else:
            cmd = ['colmap', 'sequential_matcher',
                  '--database_path', str(self.database_path),
                  '--SiftMatching.use_gpu', '1' if self.use_gpu else '0']
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    def _mapper(self) -> bool:
        cmd = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
```

### File: `gs_toolkit/core/data_validator.py`

```python
"""Data validator"""
import cv2
from pathlib import Path
import logging

logger = logging.getLogger('gs-process')

class DataValidator:
    def __init__(self, images_dir: str):
        self.images_dir = Path(images_dir)
    
    def validate(self) -> dict:
        issues = []
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        
        if len(image_files) < 10:
            issues.append(f"Too few images: {len(image_files)}")
        
        return {'valid': len(issues) == 0, 'issues': issues, 'num_images': len(image_files)}
```

### File: `gs_toolkit/core/trainer.py`

```python
"""Trainer wrapper"""
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger('gs-train')

class GaussianSplattingTrainer:
    def __init__(self, source_path: str, model_path: str, iterations: int = 30000,
                 lightweight: bool = False, **kwargs):
        self.source_path = Path(source_path)
        self.model_path = Path(model_path)
        self.iterations = iterations
        self.lightweight = lightweight
        self.params = kwargs
    
    def train(self, test_iterations=None, checkpoint_iterations=None, resume_path=None):
        gs_path = Path.home() / 'gaussian-splatting'
        if not gs_path.exists():
            logger.error("Install: git clone https://github.com/graphdeco-inria/gaussian-splatting ~/gaussian-splatting")
            return False
        
        cmd = [
            'python', str(gs_path / 'train.py'),
            '-s', str(self.source_path),
            '-m', str(self.model_path),
            '--iterations', str(self.iterations)
        ]
        
        for key, value in self.params.items():
            cmd.extend([f'--{key}', str(value)])
        
        result = subprocess.run(cmd, cwd=str(gs_path))
        return result.returncode == 0
```

### File: `gs_toolkit/core/exporter.py`

```python
"""Exporter"""
import shutil
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger('gs-export')

class ModelExporter:
    def __init__(self, model_path: str, iteration: int, output_dir: str, gpu_id: int = 0):
        self.model_path = Path(model_path)
        self.iteration = iteration
        self.output_dir = Path(output_dir)
    
    def export_ply(self) -> str:
        src = self.model_path / 'point_cloud' / f'iteration_{self.iteration}' / 'point_cloud.ply'
        dst = self.output_dir / f'model_iter_{self.iteration}.ply'
        shutil.copy(src, dst)
        return str(dst)
    
    def render_video(self, camera_config: dict, fps: int, resolution: tuple,
                     quality: str, format: str) -> str:
        gs_path = Path.home() / 'gaussian-splatting'
        
        cmd = ['python', str(gs_path / 'render.py'),
              '-m', str(self.model_path),
              '--iteration', str(self.iteration),
              '--skip_train', '--skip_test']
        
        subprocess.run(cmd, cwd=str(gs_path), check=True)
        
        render_dir = self.model_path / 'test' / f'ours_{self.iteration}' / 'renders'
        output_video = self.output_dir / f'render_iter_{self.iteration}.{format}'
        
        quality_crf = {'low': 28, 'medium': 23, 'high': 18, 'lossless': 0}
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(render_dir / '%05d.png'),
            '-c:v', 'libx264',
            '-crf', str(quality_crf[quality]),
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        return str(output_video)
```

### File: `gs_toolkit/core/camera_path.py`

```python
"""Camera path generator"""
import numpy as np
from pathlib import Path
from plyfile import PlyData

class CameraPathGenerator:
    def __init__(self, model_path: str, iteration: int):
        ply_path = Path(model_path) / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        plydata = PlyData.read(str(ply_path))
        
        positions = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
        self.scene_center = positions.mean(axis=0)
        self.scene_radius = np.linalg.norm(positions - self.scene_center, axis=1).max()
    
    def generate_orbit(self, num_frames: int, radius: float = None, height: float = 0,
                      elevation: float = 20, look_at=None):
        if radius is None:
            radius = self.scene_radius * 2.5
        if look_at is None:
            look_at = self.scene_center
        
        cameras = []
        for i in range(num_frames):
            theta = 2 * np.pi * i / num_frames
            phi = np.radians(elevation)
            
            x = radius * np.cos(theta) * np.cos(phi) + self.scene_center[0]
            y = radius * np.sin(phi) + self.scene_center[1] + height
            z = radius * np.sin(theta) * np.cos(phi) + self.scene_center[2]
            
            cameras.append({
                'position': [float(x), float(y), float(z)],
                'look_at': [float(v) for v in look_at],
                'up': [0.0, 1.0, 0.0]
            })
        
        return {'cameras': cameras, 'type': 'orbit'}
    
    def generate_spiral(self, num_frames: int, loops: float = 1.5,
                       height_range: list = None, look_at=None):
        if height_range is None:
            height_range = [-self.scene_radius, self.scene_radius]
        if look_at is None:
            look_at = self.scene_center
        
        cameras = []
        radius_base = self.scene_radius * 2.5
        
        for i in range(num_frames):
            t = i / num_frames
            theta = 2 * np.pi * loops * t
            radius = radius_base * (1 - 0.3 * t)
            height = height_range[0] + (height_range[1] - height_range[0]) * t
            
            x = radius * np.cos(theta) + self.scene_center[0]
            y = self.scene_center[1] + height
            z = radius * np.sin(theta) + self.scene_center[2]
            
            cameras.append({
                'position': [float(x), float(y), float(z)],
                'look_at': [float(v) for v in look_at],
                'up': [0.0, 1.0, 0.0]
            })
        
        return {'cameras': cameras, 'type': 'spiral'}
```

### File: `gs_toolkit/utils/logger.py`

```python
"""Logger"""
import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
```

## Step 3: Install CLI Tools

Download the three CLI files from the repository:
- `gs_toolkit/cli/gs_process.py`
- `gs_toolkit/cli/gs_train.py`
- `gs_toolkit/cli/gs_export.py`

(These were created earlier in the conversation and are ready to use)

## Step 4: Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

## Done!

Now you can use:
```bash
gs-process video --data video.mp4 --out ./data
gs-train ./data
gs-export ./data/output --ply --video
```

See README.md for complete usage guide.
