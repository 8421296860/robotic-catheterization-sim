import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    print("Initializing nnUNetPredictor...")
    results_dir = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'
    model_folder = Path(results_dir) / 'Dataset501_CarotidArtery' / 'nnUNetTrainer__nnUNetPlans__2d'
    
    if not model_folder.is_dir():
        print(f"ERROR: Model folder not found: {model_folder}")
        return

    # Suppress missing environment warnings
    os.environ['nnUNet_raw'] = 'dummy'
    os.environ['nnUNet_preprocessed'] = 'dummy'
    os.environ['nnUNet_results'] = results_dir

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    print("Model loaded successfully!")

    # Pick an image for inference
    img_path = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1419.png'
    img = Image.open(img_path)
    img_rgb = img.convert('RGB')
    img_gray = img.convert('L')
    
    img_resized_rgb = img_rgb.resize((512, 512), Image.BILINEAR)
    img_resized_gray = img_gray.resize((512, 512), Image.BILINEAR)
    
    rgb_resized = np.array(img_resized_rgb)
    gray_resized = np.array(img_resized_gray)
    
    # nnUNet typically expects shape (C, X, Y)
    # The training plans show 1 channel.
    arr = gray_resized.astype(np.float32)[np.newaxis, :, :]  # (1, 512, 512)
    
    pred_dict = predictor.predict_single_npy_array(
        input_image=arr,
        image_properties={'spacing': [999, 1, 1]},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=True,
    )
    
    probs = pred_dict[1]
    # Check probabilities shape. If binary, it might be (2, 512, 512). Artery is channel 1.
    prob = probs[1]
    
    mask = (prob > 0.45).astype(np.uint8)
    
    # Calculate Centroid
    cx, cy = 0, 0
    y_coords, x_coords = np.where(mask == 1)
    if len(y_coords) > 0:
        cy = int(np.mean(y_coords))
        cx = int(np.mean(x_coords))
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_resized)
    axes[0].set_title('Original Ultrasound Frame')
    axes[0].axis('off')
    
    axes[1].imshow(prob, cmap='inferno')
    axes[1].set_title('nnU-Net Heatmap')
    axes[1].axis('off')
    
    # Overlay mask
    overlay = rgb_resized.copy()
    overlay[mask == 1] = [0, 255, 0]  # Green mask
    axes[2].imshow(overlay)
    if cx != 0 and cy != 0:
        axes[2].plot(cx, cy, 'rx', markersize=12, markeredgewidth=3) # Red cross for centroid
    axes[2].set_title('Vessel Mask & Centroid (Tracking Target)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/admin1/Music/prep/cyint/demo_perception.png', dpi=150)
    print("Saved demo_perception.png")

if __name__ == '__main__':
    main()
