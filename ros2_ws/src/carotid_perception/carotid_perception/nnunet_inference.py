#!/usr/bin/env python3
"""
carotid_perception.nnunet_inference
=====================================
Standalone nnUNetv2 inference wrapper for use after model training.

Usage (as a script — for testing):
  python3 nnunet_inference.py \\
    --image /path/to/us_image.png \\
    --results-dir /path/to/nnunet_data/nnUNet_results

This module is also imported by perception_node.py when model_backend=nnunet.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DATASET_ID   = 501
DATASET_NAME = 'Dataset501_CarotidArtery'


def load_predictor(results_dir: str, fold: int = 0):
    """Load the trained nnUNetv2 predictor from results_dir."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch
    except ImportError as exc:
        raise ImportError(
            'nnunetv2 not installed. Install with: pip install nnunetv2'
        ) from exc

    model_folder = (
        Path(results_dir) / DATASET_NAME / 'nnUNetTrainer__nnUNetPlans__2d'
    )
    if not model_folder.is_dir():
        raise FileNotFoundError(
            f'nnUNet model folder not found: {model_folder}\n'
            f'Run training first: python train_nnunet.py --configuration 2d'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )
    print(f'nnUNet predictor loaded from {model_folder} (fold={fold}, device={device})')
    return predictor


def predict_image(predictor, bgr_image: np.ndarray) -> np.ndarray:
    """
    Run nnUNet inference on a BGR image.
    Returns a float32 probability map [0,1] same size as input.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (512, 512))
    arr = rgb_resized.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)

    seg, probs = predictor.predict_single_npy_array(
        input_image=arr,
        image_properties={'spacing': [999, 1, 1]},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=True,
    )
    # probs: (n_classes, H, W)  — class 1 = carotid artery
    prob = probs[1].astype(np.float32)
    prob = cv2.resize(prob, (bgr_image.shape[1], bgr_image.shape[0]))
    return prob


def visualise(bgr_image: np.ndarray, prob: np.ndarray, threshold: float = 0.5):
    """Show prediction overlay (blocking)."""
    binary = (prob > threshold).astype(np.uint8) * 255
    heat   = (prob * 255).clip(0, 255).astype(np.uint8)
    heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_image, 0.6, heat_c, 0.4, 0)
    # Draw contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    cv2.imshow('nnUNet: Carotid Segmentation', overlay)
    cv2.imshow('Probability Map', heat_c)
    cv2.imshow('Binary Mask', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='nnUNet carotid inference')
    parser.add_argument('--image', required=True, help='Input US image path')
    parser.add_argument(
        '--results-dir',
        default='/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results',
        help='nnUNet_results directory',
    )
    parser.add_argument('--fold', type=int, default=0, help='Model fold to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarise threshold')
    parser.add_argument('--no-display', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f'Error: cannot read image {args.image}', file=sys.stderr)
        sys.exit(1)

    predictor = load_predictor(args.results_dir, fold=args.fold)
    prob = predict_image(predictor, bgr)

    binary = prob > args.threshold
    n_pixels = binary.sum()
    total    = binary.size
    print(f'Carotid pixels: {n_pixels}/{total} ({100*n_pixels/total:.2f}%)')

    if not args.no_display:
        visualise(bgr, prob, threshold=args.threshold)


if __name__ == '__main__':
    main()
