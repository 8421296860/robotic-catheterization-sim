import cv2
import numpy as np
import torch
from pathlib import Path
import os

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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # (3, 1, H, W)
    arr = rgb.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
    arr = np.expand_dims(arr, axis=1)  # (3, 1, H, W)
    print(f"Testing array shape: {arr.shape}")
    try:
        # We must provide spacing for 3 dimensions (Z, Y, X)
        pred_dict = predictor.predict_single_npy_array(
            input_image=arr,
            image_properties={'spacing': [999, 1, 1]},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True
        )
        print("SUCCESS!")
        probs = pred_dict[1]
        print(f"Probs shape: {probs.shape}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == '__main__':
    main()
