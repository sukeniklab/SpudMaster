import os
os.chdir('..')

import sys
import xarray
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from constants import columns_names
from key_handlers import key_functions
from image_handling.imagej_manager import ImageJManager
from file_handling import generic_handlers 
from image_handling import image_functions
print(os.getcwd())

ij = ImageJManager('default').get_ij()

from datetime import datetime
from image_handling import fiji_commands, segmentation_functions
from Tracker import Tracker
from image_handling.mask_segmenters.segmentation_class import SegmentationClass
from image_handling.mask_segmenters import segmentation_algorithms
from typing import Dict, List, Callable, Optional, Union, Any
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class DirectoryStorageInformation:
    directories: List[str] = None
    current_directory: str = None
    experiment_type: str = None
    save_dir: Dict[int, str] =None
    total_images: List[str] = None
    """
    Save directory and image directories need to be the same length.
    Possibly handle single inputs by returning a single directory if none other is given.
    """

@dataclass
class ImageInfoStorage:
    """Class for keeping track of images, masks, and save names."""
    image_file: str =None
    image_name: Optional[List[str]] = None
    data_columns: Optional[Dict[str, str]] = None
    image_size_dict: Dict[str, xarray.DataArray] = field(default_factory=dict)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    mask_save_information: Optional[str] = None

    def get_image_name(self) -> List[str]:
        # Assumes split_filename returns a list of strings based on the delimiter.
        return generic_handlers.split_filename(self.image_file, '_')

    def get_image_column_variables(self) -> Dict[str, str]:
        if self.image_name is not None:
            # Create a dictionary by splitting each element of image_name by '-'
            # and using the first part as key and the second as value.
            return {item.split('-')[0]: item.split('-')[1] for item in self.image_name if '-' in item}
        else:
            return {}

class ImageProcessor:
    def __init__(self, 
                 ij: Callable, 
                 image_dir: Union[List[str], str], 
                 save_dir: Union[List[str], str]= None, 
                 properties: Union[None, str, List[str]] = None, 
                 channel: int = 0, 
                 donor_construct: str = 'mTQ2', 
                 acceptor_construct: str = 'mNG',
                 segmentation_algorithm: str = "cellpose",
                 file_type: str = 'tif', 
                 tracker_file: Any = None,
                 **kwargs):
        """
        Initialize the ImageProcessor with the given parameters.
        """
        self.ij = ij
        self.segmentation_class = SegmentationClass()
        self.directory_storage = DirectoryStorageInformation()
        self.image_data_storage = ImageInfoStorage()
        self.dataframe_storage = DataFrameInfo()
        self.file_type = file_type
        # Set directories and load image files.
        self.directory_storage.directories = image_dir
        self.directory_storage.total_images = self.__concatenate_image_files()
        
        self.plot_masks = False
        self.save_mask = False
        
        if tracker_file is not None:
            self.tracker = Tracker(tracker_file)
        else:
            self.tracker = Tracker()
        self.tracker.load_files(self.directory_storage.total_images)
        
        # Set the segmentation algorithm with any extra keyword arguments.
        self.segmentation_class.set_algorithm(segmentation_algorithm, **kwargs)
        self.segmentation_channel = channel
        
        self.donor = donor_construct
        self.acceptor = acceptor_construct
        self.dataframe = pd.DataFrame()
        self.elapsed_time = None
        
        if properties is None:
            self.properties = ['label', 'area', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std']
        elif properties == 'all':
            self.properties = ['label']
        else:
            self.properties = properties
    
    def get_directory_list(self) -> Union[List[str], str]:
        return self.directory_storage.directories

    def get_total_file_length(self) -> int:
        return len(self.directory_storage.total_images)

    def get_total_file_list(self) -> List[str]:
        return self.directory_storage.total_images
    
    def processed_time(self) -> str:
        return f"Total time to process is {self.elapsed_time} seconds"
    
    def __retrieve_files__(self, directory: str, file_type: str) -> List[str]:
        """
        Retrieve all files with the given extension from a directory.
        """
        try:
            file_list = generic_handlers.get_files_by_extension(directory, file_type)
            return file_list
        except Exception as e:
            raise RuntimeError(f"Error retrieving files from {directory}") from e
    
    def __concatenate_image_files(self) -> List[str]:
        """
        Concatenate image files from the provided directories.
        """
        directories = self.directory_storage.directories
        if isinstance(directories, str):
            directories = [directories]
            self.directory_storage.directories = directories

        image_file_list = []
        for directory in directories:
            image_file_list.extend(self.__retrieve_files__(directory, self.file_type))
        return image_file_list
    
    def __create_mask(self, image: np.ndarray) -> Union[np.ndarray, None]:
        """
        Run the segmentation algorithm on the image.
        Checks first if the image is empty.
        """
        if segmentation_functions.is_image_empty(image, pixels=25000):
            return None
        else:
            mask = self.segmentation_class.generate_mask(image)
            return mask.astype(np.uint8) if mask is not None else None 

    def __create_masks(self, image_dic: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Return a dictionary of masks from a series of timepoints.
        """
        mask_dic = {index: self.__create_mask(seg_image) for index, seg_image in image_dic.items()}
        return mask_dic
        
    def __filter_masks(self, mask_dict: Dict[int, Union[np.ndarray, None]]) -> Dict[int, np.ndarray]:
        """
        Filter out any None masks from the mask dictionary.
        """
        filtered_dict = {k: v for k, v in mask_dict.items() if v is not None}
        return filtered_dict
    
    def plot_image(self):
        """
        Stub method for plotting an image.
        """
        pass
    
    def get_image(self, image) -> str:
        """
        Open the image at the specified index and plot it.
        """
        return None
        
    def __open_image(self, image_file) -> Any:  # Replace Any with the appropriate type, e.g. xarray.DataArray
        """
        Open image using fiji_commands.
        """
        try:
            opened_image = fiji_commands.read_images_to_python(self.ij, image_file)
        except Exception as e:
            raise RuntimeError(f"Error opening image {image_file}") from e
        return opened_image
         
    def plot_mask(self):
        """
        Plot the current image mask.
        """
        pass

    def __set_DataFrame_save_info(self):
        """
        Set up data saving configuration for CSV output.
        """
        pass
    
    def __setup_segmentation_class(self):
        """
        Set up the segmentation class.
        """
        pass

    def __append_df(self):
        """
        Append dataframe information.
        """
        pass

    def save_masks(self, mask_dict):
        
    
    def get_DataFrame(self) -> pd.DataFrame:
        """
        Return the current DataFrame.
        """
        return self.dataframe

    def set_channel_names(self, channel):
        """
        Set channel names (to be implemented).
        """
        pass
    
    def __segment_image(self, image, process_all: bool = True) -> pd.DataFrame:
        """
        Process segmentation for a single image.
        Errors are caught and can be tracked.
        """
        try:
            tmp_image = self.__open_image(image)
        except Exception as e:
            # Log error if needed and return empty DataFrame.
            return pd.DataFrame()
        
        try:
            self.image_data_storage.image_dict = segmentation_functions.flatten_image(tmp_image)
        except Exception as e:
            return pd.DataFrame()
        
        
        try:
            if self.image_data_storage.construct == self.acceptor:
                seg_ch_images = segmentation_functions.get_segmentation_channel(flattened_image, 1)
            else:
                seg_ch_images = segmentation_functions.get_segmentation_channel(flattened_image, self.segmentation_channel)
        except Exception as e:
            seg_ch_images = None
        
        try: 
            output_mask = self.__create_masks(seg_ch_images)
        except Exception as e:
            output_mask = None

        try:
            filtered_mask = self.__filter_masks(output_mask) if output_mask is not None else None
        except Exception as e:
            filtered_mask = None

        if self.plot_masks:
            # Code to plot the mask (if implemented)
            pass
        
        if self.save_mask:
            # Code to save the mask (if implemented)
            pass

        try:
            image_properties = segmentation_functions.get_image_data(filtered_mask, flattened_image, self.properties)
            df = segmentation_functions.convert_dict_to_dataframe(image_properties)
        except Exception as e:
            df = pd.DataFrame()
        return df
    
    def image_segmentation(self) -> pd.DataFrame:
        """
        Perform segmentation on all images and return a consolidated DataFrame.
        """
        # Assume image_list is derived from total_images.
        self.image_data_storage.image_list = self.directory_storage.total_images
        self.tracker.display_progress()
        
        start_time = datetime.now()
        for image in self.directory_storage.total_images:
            self.image_data_storage.image_file = image
            self.image_data_storage.image_name = self.image_data_storage.get_image_name()
            self.image_data_storage.data_columns = self.image_data_storage.get_image_column_variables()
            # Optionally set construct from data_columns, e.g.:
            # self.image_data_storage.construct = self.image_data_storage.data_columns.get('construct', None)
            self.image_data_storage.image_size_dict = {key: value for key, value in image.sizes}
            df = self.__segment_image(image)
            if not df.empty:
                for key, value in self.image_data_storage.data_columns.items():
                    df[key] = value
                self.dataframe = pd.concat([self.dataframe, df], ignore_index=True)
            if self.successfully_processed:
                self.tracker.remove_file_from_list(image)
            self.tracker.display_progress()
        end_time = datetime.now()
        self.elapsed_time = (end_time - start_time).total_seconds()
        
        return self.dataframe
