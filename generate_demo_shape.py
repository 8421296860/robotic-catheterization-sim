import os
os.environ['nnUNet_raw'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    img_path = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1419.png'
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (512, 512))
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
    )
    model_folder = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results/Dataset501_CarotidArtery/nnUNetTrainer__nnUNetPlans__2d'
    predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=(0,), checkpoint_name='checkpoint_best.pth')
    
    # Try 3 channels first
    arr = rgb_resized.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
    
    # If nnunet expects pseudo-3d, let's add a Z dimension?
    # nnUNet 2D plans have transpose_forward=[0,1,2]. So the data iterators expect (C, Z, Y, X).
    arr_3d = np.expand_dims(arr, axis=1) # (3, 1, 512, 512)
    
    print("Trying shape:", arr_3d.shape)
    try:
        seg, probs = predictor.predict_single_npy_array(
            input_image=arr_3d,
            image_properties={'spacing': [999, 1, 1]},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
    except Exception as e:
        print("Failed with (3, 1, H, W), trying (1, 1, H, W)...")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (512, 512))
        arr_1d = gray_resized.astype(np.float32)[np.newaxis, np.newaxis, :, :] # (1, 1, 512, 512)
        try:
            seg, probs = predictor.predict_single_npy_array(
                input_image=arr_1d,
                image_properties={'spacing': [999, 1, 1]},
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=True,
            )
        except Exception as e2:
            print("Failed again. Trying (3, H, W) just in case...")
            seg, probs = predictor.predict_single_npy_array(
                input_image=arr,
                image_properties={'spacing': [999, 1, 1]},
                segmentation_previous_stage=None,
                output_file_truncated=None,
                save_or_return_probabilities=True,
            )
            
    print("Inference successful!")
    # probs is shape (C, Z, Y, X) -> (2, 1, 512, 512)
    if probs.ndim == 4:
        prob = probs[1, 0, :, :]
    else:
        prob = probs[1]
    
    threshold = 0.45
    binary = (prob > threshold).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_resized)
    axes[0].set_title('Original Ultrasound Frame')
    axes[0].axis('off')
    axes[1].imshow(prob, cmap='inferno')
    axes[1].set_title('nnU-Net Heatmap')
    axes[1].axis('off')
    overlay = rgb_resized.copy()
    overlay[binary == 1] = [0, 255, 0]
    axes[2].imshow(overlay)
    if cx != 0 and cy != 0:
        axes[2].plot(cx, cy, 'rx', markersize=12, markeredgewidth=3)
    axes[2].set_title('Vessel Mask & Centroid (Tracking Target)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/admin1/Music/prep/cyint/demo_perception.png', dpi=150)
    print("Saved demo_perception.png")

if __name__ == '__main__':
    main()
