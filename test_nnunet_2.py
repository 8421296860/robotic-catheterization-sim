import cv2
import numpy as np
import torch
from pathlib import Path
import os

# Set environment variables so nnUNet doesn't complain
os.environ['nnUNet_raw'] = '/tmp'
os.environ['nnUNet_preprocessed'] = '/tmp'
os.environ['nnUNet_results'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    results_dir = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'
    model_folder = Path(results_dir) / 'Dataset501_CarotidArtery' / 'nnUNetTrainer__nnUNetPlans__2d'
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    
    img_path = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1419.png'
    bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Test 1: (1, H, W)
    arr_2d = gray.astype(np.float32)[np.newaxis, ...]
    print(f"Testing 2D array shape: {arr_2d.shape}")
    try:
        predictor.predict_single_npy_array(arr_2d, {'spacing': [999, 1, 1]}, None, None, False)
        print("2D array SUCCESS!")
        return
    except Exception as e:
        print(f"2D array FAILED: {e}")
        
    # Test 2: (1, 1, H, W)
    arr_3d = gray.astype(np.float32)[np.newaxis, np.newaxis, ...]
    print(f"Testing 3D array shape: {arr_3d.shape}")
    try:
        predictor.predict_single_npy_array(arr_3d, {'spacing': [999, 1, 1]}, None, None, False)
        print("3D array SUCCESS!")
        return
    except Exception as e:
        print(f"3D array FAILED: {e}")

if __name__ == '__main__':
    main()
