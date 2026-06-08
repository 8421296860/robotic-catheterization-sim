import os

from model import carotidSegmentation

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = carotidSegmentation(model='unet_1')
image = 'data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_1677.png'
model.plot_pred(image, labels=True)