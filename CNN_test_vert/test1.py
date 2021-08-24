import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
# import nibabel as nib
import SimpleITK as sitk
import medpy as medpy
import torch
import pandas as pd


path_data = r"C:\Data\Verse2020\verse122\verse122.nii.gz"
img, H = medpy.io.load(path_data)

path_mask = r"C:\Data\Verse2020\verse122\verse122_seg.nii.gz"
mask, Hm = medpy.io.load(path_mask)


plt.figure()
plt.imshow(img[:,:,100],cmap='gray')
plt.show()
plt.imshow(mask[:,:,100],cmap='gray')
plt.show()



# fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
# ax1.imshow(montage(img), cmap ='bone')



