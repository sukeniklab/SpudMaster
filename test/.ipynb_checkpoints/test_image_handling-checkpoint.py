# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:34:27 2025

@author: patri
"""


import imagej
import numpy as np
import matplotlib.pyplot as plt

ij_path = 'D:/Fiji.app'
ij = imagej.init(ij_path, mode='interactive', add_legacy=True) 


image_1 = r'D:\olympus_images\test\test\20250124_102049_test-10-1-10-1_PK-DirectA_Image_A1-1.vsi' 
image_2 = r'D:\olympus_images\test\test\20250124_102049_test-10-1-10-1_PK-FRET, PK-FRET_Image_A1-1.vsi'


tmp = ij.io().open(image_1)
img = ij.py.from_java(tmp)
ij.py.show(img)
images = list([image_1, image_2])

tmp_array = np.array([])
for index, image in enumerate(images):
    tmp = io.imread(image)
        
    tmp_array[index] = np.concatenate((tmp_array, tmp))
    
plt.imshow(tmp[:,:])
