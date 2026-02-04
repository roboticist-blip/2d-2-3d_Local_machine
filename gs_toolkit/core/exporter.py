"""Model export and rendering with evaluation metrics"""
import shutil
import subprocess
from pathlib import Path
import json
import logging
import numpy as np
from plyfile import PlyData

logger = logging.getLogger('gs-export')

class ModelExporter:
    def __init__(self, model_path: str, iteration: int, output_dir: str, gpu_id: int = 0):
        self.model_path = Path(model_path)
        self.iteration = iteration
        self.output_dir = Path(output_dir)
        self.gpu_id = gpu_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_ply(self) -> str:
        """Copy PLY file to export directory"""
        src = self.model_path / 'point_cloud' / f'iteration_{self.iteration}' / 'point_cloud.ply'
        dst = self.output_dir / f'model_iter_{self.iteration}.ply'
        
        if not src.exists():
            raise FileNotFoundError(f"PLY not found: {src}")
        
        shutil.copy(src, dst)
        
        file_size = dst.stat().st_size / (1024 * 1024)
        logger.info(f"PLY exported: {dst} ({file_size:.2f} MB)")
        
        return str(dst)
    
    def render_video(self, camera_config: dict, fps: int, resolution: tuple,
                     quality: str, format: str) -> str:
        """Render video using camera path"""
        gs_path = Path.home() / 'gaussian-splatting'
        
        if not gs_path.exists():
            raise RuntimeError("Gaussian Splatting not found at ~/gaussian-splatting")
        
        camera_file = self.output_dir / 'camera_path.json'
        with open(camera_file, 'w') as f:
            json.dump(camera_config, f, indent=2)
        
        logger.info("Rendering frames...")
        
        cmd = [
            'python', str(gs_path / 'render.py'),
            '-m', str(self.model_path),
            '--iteration', str(self.iteration),
            '--skip_train', '--skip_test'
        ]
        
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        result = subprocess.run(cmd, cwd=str(gs_path), env=env, capture_output=True)
        
        if result.returncode != 0:
            logger.error(f"Rendering failed: {result.stderr.decode()}")
            raise RuntimeError("Rendering failed")
        
        render_dir = self.model_path / 'test' / f'ours_{self.iteration}' / 'renders'
        
        if not render_dir.exists():
            raise RuntimeError(f"Render directory not found: {render_dir}")
        
        logger.info("Converting to video...")
        
        output_video = self.output_dir / f'render_iter_{self.iteration}.{format}'
        
        quality_crf = {'low': 28, 'medium': 23, 'high': 18, 'lossless': 0}
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(render_dir / '*.png'),
            '-c:v', 'libx264',
            '-crf', str(quality_crf[quality]),
            '-pix_fmt', 'yuv420p',
            '-s', f'{resolution[0]}x{resolution[1]}',
            str(output_video)
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise RuntimeError("Video encoding failed")
        
        return str(output_video)
    
    def compute_metrics(self) -> dict:
        """Compute evaluation metrics (PSNR, SSIM, LPIPS)"""
        
        logger.info("Computing evaluation metrics...")
        
        gs_path = Path.home() / 'gaussian-splatting'
        
        metrics = {}
        
        # Try to run official metrics script
        if gs_path.exists():
            cmd = [
                'python', str(gs_path / 'metrics.py'),
                '-m', str(self.model_path)
            ]
            
            result = subprocess.run(cmd, cwd=str(gs_path), capture_output=True, text=True)
            
            # Load results if available
            results_file = self.model_path / 'results.json'
            
            if results_file.exists():
                with open(results_file) as f:
                    metrics_data = json.load(f)
                    metrics['rendering_metrics'] = metrics_data
                logger.info("✓ Rendering metrics computed (PSNR, SSIM, LPIPS)")
        
        # Compute model statistics
        ply_path = self.model_path / 'point_cloud' / f'iteration_{self.iteration}' / 'point_cloud.ply'
        
        if ply_path.exists():
            plydata = PlyData.read(str(ply_path))
            num_gaussians = len(plydata['vertex'])
            
            # Get positions
            positions = np.stack([
                plydata['vertex']['x'],
                plydata['vertex']['y'],
                plydata['vertex']['z']
            ], axis=1)
            
            # Compute bounding box
            bbox_min = positions.min(axis=0).tolist()
            bbox_max = positions.max(axis=0).tolist()
            scene_extent = float(np.linalg.norm(np.array(bbox_max) - np.array(bbox_min)))
            
            # Compute opacity statistics
            if 'opacity' in plydata['vertex'].data.dtype.names:
                opacity = np.array(plydata['vertex']['opacity'])
                opacity_values = 1 / (1 + np.exp(-opacity))
                
                opacity_stats = {
                    'mean': float(opacity_values.mean()),
                    'std': float(opacity_values.std()),
                    'num_transparent': int(np.sum(opacity_values < 0.1)),
                    'num_opaque': int(np.sum(opacity_values > 0.9))
                }
            else:
                opacity_stats = None
            
            metrics['model_statistics'] = {
                'num_gaussians': int(num_gaussians),
                'scene_extent': scene_extent,
                'bounding_box': {
                    'min': bbox_min,
                    'max': bbox_max
                },
                'file_size_mb': float(ply_path.stat().st_size / (1024 * 1024)),
                'opacity_stats': opacity_stats
            }
            
            logger.info(f"✓ Model statistics: {num_gaussians:,} Gaussians, {scene_extent:.2f} units extent")
        
        # Save metrics to file
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved: {metrics_file}")
        
        # Print summary
        self._print_metrics_summary(metrics)
        
        return metrics
    
    def _print_metrics_summary(self, metrics: dict):
        """Print metrics summary to console"""
        
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        # Rendering metrics
        if 'rendering_metrics' in metrics:
            rm = metrics['rendering_metrics']
            print("\n📊 Rendering Quality:")
            
            # Try to extract PSNR, SSIM, LPIPS
            for key, value in rm.items():
                if isinstance(value, dict):
                    if 'PSNR' in value:
                        print(f"  PSNR:  {value['PSNR']:.2f} dB")
                    if 'SSIM' in value:
                        print(f"  SSIM:  {value['SSIM']:.4f}")
                    if 'LPIPS' in value:
                        print(f"  LPIPS: {value['LPIPS']:.4f}")
        
        # Model statistics
        if 'model_statistics' in metrics:
            ms = metrics['model_statistics']
            print("\n📈 Model Statistics:")
            print(f"  Gaussians: {ms['num_gaussians']:,}")
            print(f"  File size: {ms['file_size_mb']:.2f} MB")
            print(f"  Scene extent: {ms['scene_extent']:.2f} units")
            
            if ms.get('opacity_stats'):
                ops = ms['opacity_stats']
                print(f"\n  Opacity distribution:")
                print(f"    Mean: {ops['mean']:.3f}")
                print(f"    Transparent (<0.1): {ops['num_transparent']:,} ({ops['num_transparent']/ms['num_gaussians']*100:.1f}%)")
                print(f"    Opaque (>0.9): {ops['num_opaque']:,} ({ops['num_opaque']/ms['num_gaussians']*100:.1f}%)")
        
        print("\n" + "="*60 + "\n")
