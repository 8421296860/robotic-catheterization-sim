import os
os.environ['nnUNet_raw'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/admin1/Music/prep/cyint/ros2_ws/src/carotid_perception/carotid_perception')
import nnunet_inference

def main():
    img_path = '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1419.png'
    bgr = cv2.imread(img_path)
    if bgr is None:
        print("Failed to read image")
        return
        
    predictor = nnunet_inference.load_predictor('/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results', 0)
    prob = nnunet_inference.predict_image(predictor, bgr)
    
    threshold = 0.45
    binary = (prob > threshold).astype(np.uint8)
    
    # Calculate Centroid
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title('Original Ultrasound Frame')
    axes[0].axis('off')
    
    axes[1].imshow(prob, cmap='inferno')
    axes[1].set_title('nnU-Net Heatmap')
    axes[1].axis('off')
    
    # Overlay mask
    overlay = rgb.copy()
    overlay[binary == 1] = [0, 255, 0]  # Green mask
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
