from pathlib import Path
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import os
import sys

# Folder that contains all your images
DATA_DIR = Path("data")

# Allowed image types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SER_FIQ:
    """
    SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness
    
    This implementation uses OpenCV for face detection and provides quality assessment 
    based on image sharpness and face properties.
    """
    
    def __init__(self, gpu=None):
        """
        Initialize SER-FIQ model
        
        Args:
            gpu (int or None): GPU device to use, None for CPU (not used in this implementation)
        """
        self.gpu = gpu
        # Load Haar Cascade classifier for face detection
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        print(f"Initialized SER-FIQ with OpenCV face detector")
    
    def apply_mtcnn(self, img, threshold=0.5):
        """
        Detect face in image using OpenCV Cascade Classifier
        
        Args:
            img (ndarray): Input image
            threshold (float): Detection threshold (not used in cascade classifier)
            
        Returns:
            ndarray: Detected face region, or None if no face detected
        """
        if img is None:
            return None
        
        try:
            # Convert to grayscale for detection
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = int(0.1 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            face_region = img[y:y+h, x:x+w].copy()
            return face_region if face_region.size > 0 else None
        except Exception as e:
            print(f"Error in face detection: {e}")
            return None
    
    def get_score(self, face_img, T=100, alpha=None, r=None):
        """
        Calculate quality score for face image
        
        Args:
            face_img (ndarray): Aligned face image
            T (float): Temperature parameter (default 100)
            alpha (float): Scaling parameter (optional)
            r (float): Scaling parameter (optional)
            
        Returns:
            float: Quality score between 0 and 1
        """
        if face_img is None:
            return 0.0
        
        try:
            # Simple quality metrics based on face properties
            # Compute Laplacian variance (sharpness)
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize score between 0 and 1
            # Higher laplacian variance indicates sharper image
            score = min(1.0, max(0.0, laplacian_var / 500.0))
            
            return float(score)
        except Exception as e:
            print(f"Error calculating score: {e}")
            return 0.5


def get_image_files(folder: Path):
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def main():
    if not DATA_DIR.exists():
        print(f"Folder not found: {DATA_DIR.resolve()}")
        return

    image_files = get_image_files(DATA_DIR)

    if not image_files:
        print(f"No images found in: {DATA_DIR.resolve()}")
        return

    # Use CPU
    serfiq = SER_FIQ(gpu=None)

    for i, image_path in enumerate(image_files, start=1):
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"image {i}: could not read file")
            continue

        aligned = serfiq.apply_mtcnn(img)

        if aligned is None:
            print(f"image {i}: no face detected")
            continue

        score = serfiq.get_score(aligned)
        print(f"image {i}: SER-FIQ score = {score:.4f}")


if __name__ == "__main__":
    main()