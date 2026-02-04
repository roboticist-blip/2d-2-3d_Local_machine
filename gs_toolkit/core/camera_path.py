"""Camera path generator"""
import numpy as np
from pathlib import Path
from plyfile import PlyData

class CameraPathGenerator:
    def __init__(self, model_path: str, iteration: int):
        ply_path = Path(model_path) / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found: {ply_path}")
        
        plydata = PlyData.read(str(ply_path))
        
        positions = np.stack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ], axis=1)
        
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
    
    def generate_linear(self, num_frames: int, start_pos: list, end_pos: list, look_at=None):
        if look_at is None:
            look_at = self.scene_center
        
        start = np.array(start_pos)
        end = np.array(end_pos)
        
        cameras = []
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            pos = start + (end - start) * t
            
            cameras.append({
                'position': pos.tolist(),
                'look_at': [float(v) for v in look_at],
                'up': [0.0, 1.0, 0.0]
            })
        
        return {'cameras': cameras, 'type': 'linear'}
