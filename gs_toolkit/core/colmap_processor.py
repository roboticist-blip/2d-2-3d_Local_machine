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
            raise RuntimeError("COLMAP not found. Install with: sudo apt-get install colmap")
    
    def process(self) -> bool:
        try:
            logger.info("Running COLMAP feature extraction...")
            if not self._feature_extraction():
                return False
            logger.info("Running COLMAP feature matching...")
            if not self._feature_matching():
                return False
            logger.info("Running COLMAP mapper...")
            if not self._mapper():
                return False
            logger.info("COLMAP processing complete")
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
            logger.info("Lightweight mode: using medium quality")
        
        preset = quality_presets.get(self.quality, quality_presets['high'])
        
        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.single_camera', '1' if self.single_camera else '0',
            '--ImageReader.camera_model', 'OPENCV',
            '--SiftExtraction.use_gpu', '1' if self.use_gpu else '0',
            '--SiftExtraction.max_image_size', str(preset['max_image_size']),
            '--SiftExtraction.max_num_features', str(preset['max_num_features'])
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Feature extraction failed: {result.stderr}")
            return False
        return True
    
    def _feature_matching(self) -> bool:
        matcher = 'sequential' if self.lightweight else 'exhaustive'
        
        if matcher == 'exhaustive':
            cmd = ['colmap', 'exhaustive_matcher',
                  '--database_path', str(self.database_path),
                  '--SiftMatching.use_gpu', '1' if self.use_gpu else '0']
        else:
            cmd = ['colmap', 'sequential_matcher',
                  '--database_path', str(self.database_path),
                  '--SiftMatching.use_gpu', '1' if self.use_gpu else '0',
                  '--SequentialMatching.overlap', '10']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Feature matching failed: {result.stderr}")
            return False
        return True
    
    def _mapper(self) -> bool:
        cmd = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir)
        ]
        
        if self.lightweight:
            cmd.extend([
                '--Mapper.ba_global_max_num_iterations', '50',
                '--Mapper.ba_local_max_num_iterations', '25'
            ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Mapper failed: {result.stderr}")
            return False
        return True
