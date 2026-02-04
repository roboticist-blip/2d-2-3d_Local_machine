#!/usr/bin/env python3
"""gs-export: Export model and render videos"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.exporter import ModelExporter
from core.camera_path import CameraPathGenerator
from utils.logger import setup_logger
import json

logger = setup_logger('gs-export')

def main():
    parser = argparse.ArgumentParser(description='Export 3D Gaussian Splatting model')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--ply', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--camera-path', choices=['orbit', 'spiral', 'linear', 'custom'], default='orbit')
    parser.add_argument('--camera-config', help='Path to custom camera JSON')
    parser.add_argument('--num-frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080])
    parser.add_argument('--orbit-radius', type=float, default=None)
    parser.add_argument('--orbit-height', type=float, default=0.0)
    parser.add_argument('--orbit-elevation', type=float, default=20.0)
    parser.add_argument('--spiral-loops', type=float, default=1.5)
    parser.add_argument('--spiral-height-range', type=float, nargs=2, default=[-0.5, 0.5])
    parser.add_argument('--linear-start', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--linear-end', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--look-at', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--video-quality', choices=['low', 'medium', 'high', 'lossless'], default='high')
    parser.add_argument('--video-format', choices=['mp4', 'mov', 'avi'], default='mp4')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Default: export both
    if not args.ply and not args.video:
        args.ply = True
        args.video = True
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else (model_path / 'exports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ply_path = model_path / 'point_cloud' / f'iteration_{args.iteration}' / 'point_cloud.ply'
    if not ply_path.exists():
        logger.error(f"PLY not found at iteration {args.iteration}: {ply_path}")
        return 1
    
    logger.info(f"Exporting model: {model_path}")
    
    try:
        exporter = ModelExporter(str(model_path), args.iteration, str(output_dir), args.gpu)
        
        exports = []
        
        # Export PLY
        if args.ply:
            logger.info("Exporting PLY...")
            ply_out = exporter.export_ply()
            exports.append(ply_out)
        
        # Render video
        if args.video:
            logger.info(f"Rendering video with {args.camera_path} camera path...")
            
            # Generate camera path
            if args.camera_path == 'custom' and args.camera_config:
                with open(args.camera_config) as f:
                    camera_config = json.load(f)
            else:
                camera_gen = CameraPathGenerator(str(model_path), args.iteration)
                
                if args.camera_path == 'orbit':
                    camera_config = camera_gen.generate_orbit(
                        num_frames=args.num_frames,
                        radius=args.orbit_radius,
                        height=args.orbit_height,
                        elevation=args.orbit_elevation,
                        look_at=args.look_at
                    )
                elif args.camera_path == 'spiral':
                    camera_config = camera_gen.generate_spiral(
                        num_frames=args.num_frames,
                        loops=args.spiral_loops,
                        height_range=args.spiral_height_range,
                        look_at=args.look_at
                    )
                elif args.camera_path == 'linear':
                    if not args.linear_start or not args.linear_end:
                        logger.error("Linear path requires --linear-start and --linear-end")
                        return 1
                    camera_config = camera_gen.generate_linear(
                        num_frames=args.num_frames,
                        start_pos=args.linear_start,
                        end_pos=args.linear_end,
                        look_at=args.look_at
                    )
            
            # Save camera path
            camera_file = output_dir / f'camera_{args.camera_path}.json'
            with open(camera_file, 'w') as f:
                json.dump(camera_config, f, indent=2)
            logger.info(f"Camera path saved: {camera_file}")
            
            # Render
            video_out = exporter.render_video(
                camera_config=camera_config,
                fps=args.fps,
                resolution=tuple(args.resolution),
                quality=args.video_quality,
                format=args.video_format
            )
            exports.append(video_out)
        
        # Compute evaluation metrics
        logger.info("\nComputing evaluation metrics...")
        metrics = exporter.compute_metrics()
        
        logger.info("\n✓ Export complete!")
        for f in exports:
            logger.info(f"  {f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
