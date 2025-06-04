import os
import sys
import xarray
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import skimage as ski
from file_handling.generic_handlers import dir_file_split_len
from image_handling.imagej_manager import ImageJManager
from analysis_class import ImageProcessor
from image_handler_class import ImageAlignmentAndTifConversion
from key_class import Key

ij = ImageJManager('default').get_ij()

segmenter = ImageProcessor(ij, r'D:/Patrick/Infection/save_files/Tif/aligned_images/',  tracker_file=r'D:\Patrick\tracking_jsons\heat_tracker.json', dataframe_file=r'D:\Patrick\dataframes\new_heat-blast_dataframe.pkl')
df = segmenter.image_segmentation()
df.to_csv(r'D:\Patrick\dataframes\heat-blast_dataframe.csv')