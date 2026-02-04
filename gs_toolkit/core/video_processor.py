"""Video processing module"""
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
        
        logger.info(f"Video: {orig_fps:.2f} fps, {total_frames} frames")
        
        if fps is None:
            fps = orig_fps
        
        frame_interval = max(1, int(orig_fps / fps))
        if max_frames:
            frame_interval = max(frame_interval, total_frames // max_frames)
        
        if self.lightweight and quality > 90:
            quality = 90
        
        logger.info(f"Extracting every {frame_interval} frame(s)")
        
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
            
            if frame_count % 100 == 0:
                logger.info(f"Progress: {frame_count}/{total_frames}, extracted {extracted_count}")
        
        cap.release()
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Extracted {extracted_count} frames to {self.output_dir}")
        return extracted_count
