#!/usr/bin/env python3
"""gs-train: Train 3D Gaussian Splatting model"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trainer import GaussianSplattingTrainer
from utils.logger import setup_logger
import json

logger = setup_logger('gs-train')

def main():
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting model')
    parser.add_argument('data_path', help='Path to processed data')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--test-iterations', type=int, nargs='+', default=[7000, 15000, 30000])
    parser.add_argument('--checkpoint-iterations', type=int, nargs='+', default=[7000, 15000, 30000])
    parser.add_argument('--lightweight', action='store_true')
    parser.add_argument('--position-lr-init', type=float, default=0.00016)
    parser.add_argument('--position-lr-final', type=float, default=0.0000016)
    parser.add_argument('--feature-lr', type=float, default=0.0025)
    parser.add_argument('--opacity-lr', type=float, default=0.05)
    parser.add_argument('--scaling-lr', type=float, default=0.005)
    parser.add_argument('--rotation-lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        return 1
    
    # Check for config
    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if config.get('lightweight') and not args.lightweight:
                args.lightweight = True
                logger.info("Enabling lightweight mode from config")
    
    model_path = Path(args.model_path) if args.model_path else (data_path / 'output')
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training: {data_path} -> {model_path}")
    logger.info(f"Iterations: {args.iterations}, Lightweight: {args.lightweight}")
    
    try:
        trainer = GaussianSplattingTrainer(
            source_path=str(data_path),
            model_path=str(model_path),
            iterations=args.iterations,
            lightweight=args.lightweight,
            position_lr_init=args.position_lr_init,
            position_lr_final=args.position_lr_final,
            feature_lr=args.feature_lr,
            opacity_lr=args.opacity_lr,
            scaling_lr=args.scaling_lr,
            rotation_lr=args.rotation_lr,
            gpu_id=args.gpu,
            quiet=args.quiet
        )
        
        success = trainer.train(
            test_iterations=args.test_iterations,
            checkpoint_iterations=args.checkpoint_iterations
        )
        
        if not success:
            logger.error("Training failed")
            return 1
        
        final_ply = model_path / 'point_cloud' / f'iteration_{args.iterations}' / 'point_cloud.ply'
        
        if final_ply.exists():
            size_mb = final_ply.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Training complete! Model: {final_ply} ({size_mb:.1f} MB)")
            logger.info(f"Next: gs-export {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
