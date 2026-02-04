#!/usr/bin/env python3
"""gs-process: Data preprocessing for 3D Gaussian Splatting"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.video_processor import VideoProcessor
from core.colmap_processor import ColmapProcessor
from core.data_validator import DataValidator
from utils.logger import setup_logger
import json

logger = setup_logger('gs-process')

def main():
    parser = argparse.ArgumentParser(description='Process video/images for 3D Gaussian Splatting')
    parser.add_argument('input_type', choices=['video', 'images'])
    parser.add_argument('--data', required=True, help='Input video/image folder')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--resolution', type=int, nargs=2, metavar=('W', 'H'))
    parser.add_argument('--quality', type=int, default=95)
    parser.add_argument('--skip-colmap', action='store_true')
    parser.add_argument('--colmap-quality', choices=['low', 'medium', 'high', 'extreme'], default='high')
    parser.add_argument('--lightweight', action='store_true')
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {args.input_type}: {data_path} -> {output_path}")
    
    try:
        # Extract frames
        if args.input_type == 'video':
            processor = VideoProcessor(str(data_path), str(output_path / 'images'), args.lightweight)
            num_frames = processor.extract_frames(
                fps=args.fps,
                max_frames=args.max_frames,
                resolution=tuple(args.resolution) if args.resolution else None,
                quality=args.quality
            )
            images_dir = output_path / 'images'
        else:
            import shutil
            images_dir = output_path / 'images'
            images_dir.mkdir(exist_ok=True)
            for img in data_path.glob("*.jpg") + list(data_path.glob("*.png")):
                shutil.copy(img, images_dir / img.name)
            num_frames = len(list(images_dir.glob("*")))
        
        # Validate
        validator = DataValidator(str(images_dir))
        report = validator.validate()
        if not report['valid']:
            for issue in report['issues']:
                logger.warning(issue)
        
        # COLMAP
        if not args.skip_colmap:
            colmap = ColmapProcessor(
                str(images_dir), str(output_path),
                quality=args.colmap_quality,
                lightweight=args.lightweight
            )
            if not colmap.process():
                logger.error("COLMAP failed")
                return 1
        
        # Save config
        config = {
            'source_path': str(output_path),
            'images': 'images',
            'sparse_model': 'sparse/0',
            'num_images': num_frames,
            'lightweight': args.lightweight
        }
        
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Processing complete! Ready for: gs-train {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
