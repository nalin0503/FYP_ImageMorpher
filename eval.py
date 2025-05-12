# -*- coding: utf-8 -*-
"""
Metamorph Evaluation Script

This script evaluates image morphing sequences from the Metamorph framework
by computing Fréchet Inception Distance (FID), Perceptual Path Length (PPL),
frame rate, and other performance metrics.

Example usage:
    python eval.py --video path/to/video.mp4 --real-dir path/to/real/images --output results.json
    python eval.py --frames-dir path/to/frames --real-dir path/to/real/images --output results.json
    python eval.py --compare config.json --real-dir path/to/real/images --output comparison.csv

Json config example for comparison:

{
  "DiffMorpher": {
    "frames_dir": "/path/to/diffmorpher/frames",
    "runtime": 402.8
  },
  "DDIM_SLERP": {
    "frames_dir": "/path/to/ddim_slerp/frames",
    "runtime": 395.2
  },
  "Metamorph_Max_Quality": {
    "video_path": "/path/to/metamorph_max/output.mp4",
    "runtime": 407.4
  },
  "Metamorph_Medium": {
    "frames_dir": "/path/to/metamorph_medium/frames",
    "runtime": 398.6
  },
  "Metamorph_LCM_LoRA": {
    "video_path": "/path/to/metamorph_lcm/output.mp4",
    "runtime": 403.2
  }
}

"""
import os
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
from scipy import linalg
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metamorph_eval")


class InceptionV3FeatureExtractor(nn.Module):
    """
    Inception V3 network for feature extraction, used in FID and PPL calculations.
    
    This implementation extracts features from the pool3 layer (2048-dim) of
    Inception V3 pretrained on ImageNet, which is standard for FID calculation.
    """
    def __init__(self, device: str = "cuda"):
        """
        Initialize the Inception V3 feature extractor.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e
        )
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.device = device
        self.to(device)
        self.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Batch of preprocessed images [B, 3, 299, 299]
            
        Returns:
            features: Extracted features with shape [B, 2048]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(x, 1)


class MetamorphEvaluator:
    """
    Evaluates image morphing sequences from the Metamorph framework.
    
    Computes the following metrics:
    - Fréchet Inception Distance (FID): Measure of image fidelity/quality
    - Perceptual Path Length (PPL): Measure of morphing smoothness
    - Runtime: Processing time for generating morphing sequences
    - Frame rate: Effective frames per second
    """
    
    def __init__(
        self, 
        real_images_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator with paths and models.
        
        Args:
            real_images_dir: Directory containing real images for FID calculation
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        self.real_images_dir = real_images_dir
        self.device = device
        
        # Initialize feature extractor for FID and PPL calculations
        logger.info(f"Initializing Inception V3 on {device}")
        self.feature_extractor = InceptionV3FeatureExtractor(device)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        # Cache for real images statistics (for FID)
        self.real_mean: Optional[np.ndarray] = None
        self.real_cov: Optional[np.ndarray] = None
        
    def precompute_real_statistics(self) -> None:
        """
        Precompute statistics of real images for FID calculation.
        
        Extracts features from real images and computes their mean and covariance.
        Results are cached for repeated FID evaluations.
        """
        logger.info(f"Computing statistics for real images in {self.real_images_dir}")
        
        # Get all image paths
        real_image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            real_image_paths.extend(list(Path(self.real_images_dir).glob(ext)))
        
        if not real_image_paths:
            raise ValueError(f"No images found in {self.real_images_dir}")
            
        logger.info(f"Found {len(real_image_paths)} real images")
        
        # Extract features in batches
        batch_size = 32
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(real_image_paths), batch_size)):
                batch_paths = real_image_paths[i:i+batch_size]
                batch_images = []
                
                for path in batch_paths:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                    
                if not batch_images:
                    continue
                    
                batch_tensor = torch.stack(batch_images).to(self.device)
                features = self.feature_extractor(batch_tensor)
                all_features.append(features.cpu().numpy())
                
        if not all_features:
            raise ValueError("Failed to extract features from real images")
            
        all_features = np.concatenate(all_features, axis=0)
        self.real_mean = np.mean(all_features, axis=0)
        self.real_cov = np.cov(all_features, rowvar=False)
        
        logger.info(f"Computed statistics for {all_features.shape[0]} real images")
        
    def extract_features_from_frames(
        self,
        frames_dir: str
    ) -> np.ndarray:
        """
        Extract features from all frames in a directory.
        
        Args:
            frames_dir: Directory containing image frames
            
        Returns:
            features: NumPy array of extracted features [N, 2048]
        """
        # Get all image paths
        frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_paths.extend(list(Path(frames_dir).glob(ext)))
            
        # Sort frames by name (assuming sequential naming)
        frame_paths.sort()
        
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_dir}")
            
        logger.info(f"Extracting features from {len(frame_paths)} frames")
        
        # Extract features in batches
        batch_size = 32
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(frame_paths), batch_size)):
                batch_paths = frame_paths[i:i+batch_size]
                batch_images = []
                
                for path in batch_paths:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                    
                if not batch_images:
                    continue
                    
                batch_tensor = torch.stack(batch_images).to(self.device)
                features = self.feature_extractor(batch_tensor)
                all_features.append(features.cpu().numpy())
                
        if not all_features:
            raise ValueError("Failed to extract features from frames")
            
        return np.concatenate(all_features, axis=0)
        
    def extract_features_from_video(
        self,
        video_path: str,
        sample_rate: int = 1
    ) -> np.ndarray:
        """
        Extract features from frames in a video file.
        
        Args:
            video_path: Path to the video file
            sample_rate: Process every Nth frame (default: 1, process all frames)
            
        Returns:
            features: NumPy array of extracted features [N, 2048]
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
            
        logger.info(f"Extracting features from video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video contains {frame_count} frames")
        
        # Extract features in batches
        batch_size = 32
        batch_images = []
        all_features = []
        frame_idx = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img_tensor = self.preprocess(img)
                    batch_images.append(img_tensor)
                    
                    if len(batch_images) == batch_size:
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        features = self.feature_extractor(batch_tensor)
                        all_features.append(features.cpu().numpy())
                        batch_images = []
                        
                frame_idx += 1
                
            # Process remaining images
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                features = self.feature_extractor(batch_tensor)
                all_features.append(features.cpu().numpy())
                
        cap.release()
        
        if not all_features:
            raise ValueError("Failed to extract features from video")
            
        return np.concatenate(all_features, axis=0)
        
    def calculate_fid(self, features: np.ndarray) -> float:
        """
        Calculate Fréchet Inception Distance (FID) between real and generated images.
        
        Lower FID indicates higher fidelity and better quality of generated images.
        
        Args:
            features: Features extracted from generated images
            
        Returns:
            fid_score: The calculated FID score
        """
        if self.real_mean is None or self.real_cov is None:
            self.precompute_real_statistics()
            
        gen_mean = np.mean(features, axis=0)
        gen_cov = np.cov(features, rowvar=False)
        
        # Calculate FID
        mu1, sigma1 = self.real_mean, self.real_cov
        mu2, sigma2 = gen_mean, gen_cov
        
        # Means difference term
        mean_diff_sq = np.sum((mu1 - mu2) ** 2)
        
        # Covariance term
        # Calculate matrix square root of product
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Ensure covmean is real
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = mean_diff_sq + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
        
    def calculate_ppl(self, features: np.ndarray) -> float:
        """
        Calculate Perceptual Path Length (PPL) for a morphing sequence.
        
        Lower PPL indicates smoother transitions between frames.
        
        Args:
            features: Features extracted from sequential frames
            
        Returns:
            ppl_score: The calculated PPL score
        """
        if features.shape[0] < 2:
            raise ValueError("Need at least 2 frames to calculate PPL")
            
        # Calculate Euclidean distances between consecutive frames in feature space
        feature_diffs = np.sqrt(np.sum((features[1:] - features[:-1]) ** 2, axis=1))
        
        # PPL is the average of these differences
        ppl = float(np.mean(feature_diffs))
        
        return ppl
        
    def evaluate_frames(self, frames_dir: str) -> Dict[str, float]:
        """
        Evaluate a sequence of image frames.
        
        Args:
            frames_dir: Directory containing image frames
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        start_time = time.time()
        
        # Extract features from frames
        features = self.extract_features_from_frames(frames_dir)
        num_frames = features.shape[0]
        
        # Calculate metrics
        fid = self.calculate_fid(features)
        ppl = self.calculate_ppl(features)
        
        eval_time = time.time() - start_time
        
        return {
            "fid": fid,
            "ppl": ppl,
            "num_frames": num_frames,
            "eval_time": eval_time
        }
        
    def evaluate_video(
        self,
        video_path: str,
        runtime: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate a video file containing a morphing sequence.
        
        Args:
            video_path: Path to the video file
            runtime: Optional runtime in seconds (if known from logs)
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        start_time = time.time()
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Extract features from video frames
        features = self.extract_features_from_video(video_path)
        
        # Calculate metrics
        fid = self.calculate_fid(features)
        ppl = self.calculate_ppl(features)
        
        # Use provided runtime or None
        effective_fps = frame_count / runtime if runtime else None
        
        eval_time = time.time() - start_time
        
        return {
            "fid": fid,
            "ppl": ppl,
            "num_frames": frame_count,
            "fps": fps,
            "duration": duration,
            "runtime": runtime,
            "effective_fps": effective_fps,
            "eval_time": eval_time
        }
        
    def run_comparison(
        self,
        methods: Dict[str, Dict[str, Any]],
        output_file: str
    ) -> None:
        """
        Run comparison between different morphing methods.
        
        Args:
            methods: Dictionary mapping method names to their parameters
                Each method should have either 'video_path' or 'frames_dir',
                and optionally 'runtime' if known
            output_file: Path to save results in CSV format
        """
        results = []
        
        for method_name, params in methods.items():
            logger.info(f"Evaluating method: {method_name}")
            
            try:
                if 'video_path' in params:
                    metrics = self.evaluate_video(
                        params['video_path'],
                        params.get('runtime')
                    )
                elif 'frames_dir' in params:
                    metrics = self.evaluate_frames(params['frames_dir'])
                    if 'runtime' in params:
                        metrics['runtime'] = params['runtime']
                        if metrics['num_frames'] > 0 and params['runtime'] > 0:
                            metrics['effective_fps'] = metrics['num_frames'] / params['runtime']
                else:
                    logger.error(f"Method {method_name} must have either video_path or frames_dir")
                    continue
                    
                # Add method name to metrics
                metrics['method'] = method_name
                results.append(metrics)
                
                logger.info(f"Results for {method_name}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")
                        
            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}")
                
        # Save results to CSV
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.warning("No results to save")
            

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metamorph Evaluation Tool")
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode')
    
    # Video evaluation
    video_parser = subparsers.add_parser('video', help='Evaluate a video file')
    video_parser.add_argument('--video', required=True, help='Path to the video file')
    video_parser.add_argument('--real-dir', required=True, help='Directory with real images')
    video_parser.add_argument('--runtime', type=float, help='Runtime in seconds (optional)')
    video_parser.add_argument('--output', default='metamorph_video_eval.json', 
                             help='Output file path for results (JSON)')
    
    # Frames evaluation
    frames_parser = subparsers.add_parser('frames', help='Evaluate image frames')
    frames_parser.add_argument('--frames-dir', required=True, help='Directory with frame images')
    frames_parser.add_argument('--real-dir', required=True, help='Directory with real images')
    frames_parser.add_argument('--runtime', type=float, help='Runtime in seconds (optional)')
    frames_parser.add_argument('--output', default='metamorph_frames_eval.json',
                             help='Output file path for results (JSON)')
    
    # Comparison evaluation
    compare_parser = subparsers.add_parser('compare', help='Compare multiple methods')
    compare_parser.add_argument('--config', required=True, help='JSON config file for comparison')
    compare_parser.add_argument('--real-dir', required=True, help='Directory with real images')
    compare_parser.add_argument('--output', default='metamorph_comparison.csv',
                              help='Output file path for results (CSV)')
    
    # Device selection
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Initialize evaluator
    evaluator = MetamorphEvaluator(
        real_images_dir=args.real_dir,
        device=args.device
    )
    
    # Run evaluation based on mode
    if args.mode == 'video':
        logger.info(f"Evaluating video: {args.video}")
        metrics = evaluator.evaluate_video(args.video, args.runtime)
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output}")
        
    elif args.mode == 'frames':
        logger.info(f"Evaluating frames in: {args.frames_dir}")
        metrics = evaluator.evaluate_frames(args.frames_dir)
        
        # Add runtime if provided
        if args.runtime:
            metrics['runtime'] = args.runtime
            if metrics['num_frames'] > 0:
                metrics['effective_fps'] = metrics['num_frames'] / args.runtime
                
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output}")
        
    elif args.mode == 'compare':
        logger.info(f"Running comparison using config: {args.config}")
        
        # Load config
        import json
        with open(args.config, 'r') as f:
            methods = json.load(f)
            
        evaluator.run_comparison(methods, args.output)
        
    else:
        logger.error("Invalid mode. Use 'video', 'frames', or 'compare'")
        return 1
        
    return 0


if __name__ == "__main__":
    main()