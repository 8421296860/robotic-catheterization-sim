import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    print("Initializing nnUNetPredictor...")
    results_dir = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'
    model_folder = Path(results_dir) / 'Dataset501_CarotidArtery' / 'nnUNetTrainer__nnUNetPlans__2d'
    
    if not model_folder.is_dir():
        print(f"ERROR: Model folder not found: {model_folder}")
        return

    try:
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
        print("Loading checkpoint...")
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=(0,),
            checkpoint_name='checkpoint_best.pth',
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # Pick an image for inference
    img_path = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1419.png'
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (512, 512))
    
    arr = rgb_resized.astype(np.float32).transpose(2, 0, 1)  # (3,512,512)
    print(f"Input array shape: {arr.shape}")
    
    try:
        pred_dict = predictor.predict_single_npy_array(
            input_image=arr,
            image_properties={'spacing': [999, 1, 1]},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        # pred_dict is (seg, probs)
        probs = pred_dict[1]
        print(f"Probabilities shape: {probs.shape}")
        
        prob = probs[1]  # artery channel
        max_prob = prob.max()
        print(f"Maximum probability of artery: {max_prob:.4f}")
        
        if max_prob > 0.45:
            print("VESSEL DETECTED!")
        else:
            print("NO VESSEL DETECTED.")
            
    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
