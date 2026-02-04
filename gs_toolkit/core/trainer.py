"""Trainer wrapper"""
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger('gs-train')

class GaussianSplattingTrainer:
    def __init__(self, source_path: str, model_path: str, iterations: int = 30000,
                 lightweight: bool = False, target_gaussians: int = None, **kwargs):
        self.source_path = Path(source_path)
        self.model_path = Path(model_path)
        self.iterations = iterations
        self.lightweight = lightweight
        self.target_gaussians = target_gaussians
        self.params = kwargs
        self.quiet = kwargs.get('quiet', False)
        self.gpu_id = kwargs.get('gpu_id', 0)
    
    def train(self, test_iterations=None, checkpoint_iterations=None, resume_path=None):
        gs_path = Path.home() / 'gaussian-splatting'
        if not gs_path.exists():
            logger.error("Gaussian Splatting not found!")
            logger.error("Install: git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive ~/gaussian-splatting")
            logger.error("Then: cd ~/gaussian-splatting && pip install submodules/diff-gaussian-rasterization submodules/simple-knn")
            return False
        
        cmd = [
            'python', str(gs_path / 'train.py'),
            '-s', str(self.source_path),
            '-m', str(self.model_path),
            '--iterations', str(self.iterations)
        ]
        
        for key, value in self.params.items():
            if key not in ['quiet', 'gpu_id', 'target_gaussians']:
                cmd.extend([f'--{key}', str(value)])
        
        if test_iterations:
            cmd.extend(['--test_iterations'] + [str(i) for i in test_iterations])
        
        if checkpoint_iterations:
            cmd.extend(['--checkpoint_iterations'] + [str(i) for i in checkpoint_iterations])
        
        if self.lightweight:
            cmd.extend([
                '--densify_grad_threshold', '0.0003',
                '--densification_interval', '150',
                '--opacity_reset_interval', '4000'
            ])
            logger.info("Lightweight mode enabled")
        
        logger.info(f"Starting training with {self.iterations} iterations...")
        logger.info(f"Command: {' '.join(cmd[:5])}...")
        
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        result = subprocess.run(cmd, cwd=str(gs_path), env=env)
        
        return result.returncode == 0
