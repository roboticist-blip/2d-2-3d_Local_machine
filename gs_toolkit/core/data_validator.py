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
            issues.append(f"Too few images: {len(image_files)} (recommend 50+)")
        
        for img_path in image_files[:5]:
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append(f"Cannot read: {img_path.name}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur < 50:
                issues.append(f"Possibly blurry: {img_path.name}")
        
        return {'valid': len(issues) == 0, 'issues': issues, 'num_images': len(image_files)}
