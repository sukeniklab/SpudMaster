import os
import shutil
import json
import sys
import xarray
#import rasterio
import gc
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from constants import columns_names
from key_handlers import key_functions

try:
    from IPython.display import clear_output
except:
    pass
    
import time
from file_handling import generic_handlers 
from image_handling import image_functions
from datetime import datetime
from image_handling import fiji_commands, segmentation_functions
from Tracker import Tracker
from image_handling.mask_segmenters.segmentation_class import SegmentationClass
from image_handling.mask_segmenters import segmentation_algorithms
from data_storage_classes.data_classes import DirectoryStorageInformation, ImageInfoStorage, DataFrameHolder
from typing import Dict, List, Callable, Optional, Union, Any
import pandas as pd

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
                 dataframe_file: str=  None,
                 experiment = 'nonosmotic',
                 **kwargs):
        """
        Initialize the ImageProcessor with the given parameters.
        """
        self.ij = ij
        self.segmentation_class = SegmentationClass()
        self.segmentation_class.set_algorithm(segmentation_algorithm, **kwargs)
        self.run_time = 0
        self.directory_storage = DirectoryStorageInformation()
        self.image_data_storage = ImageInfoStorage()
        if dataframe_file == None:
            self.dataframe_storage = DataFrameHolder()
        else:
            self.dataframe_storage = DataFrameHolder(storage_file = dataframe_file)
        self.file_type = file_type
        # Set directories and load image files.
        self.directory_storage.directories = image_dir
        
        if save_dir != None:
            self.directory_storage.save_dir = save_dir
            
        self.plot_masks = False
        self.save_mask = False
        self.experiment = experiment

        if tracker_file != None:
            self.tracker = Tracker(tracker_file)
        else:
            cwd = os.getcwd()
            self.tracker = Tracker(os.path.join(cwd, 'tracking_file.json'))
        
        
        
        
        if not self.tracker.current_file_list():
            self.directory_storage.total_images = self.__concatenate_image_files()    
            self.tracker.load_files(self.directory_storage.total_images)
        else:
            self.directory_storage.total_images = self.tracker.current_file_list()
        # Set the segmentation algorithm with any extra keyword arguments.
        
        self.segmentation_channel = channel
        self.donor = donor_construct
        self.acceptor = acceptor_construct
        self.elapsed_time = 0
        
        if properties is None:
            self.properties = ['label', 'area', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std']
        elif properties == 'all':
            self.properties = ['label']
        else:
            self.properties = properties
    
    def total_time(self):
        return self.elapsed_time
    
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
        
        if isinstance(self.directory_storage.directories, str):
            self.directory_storage.directories= [self.directory_storage.directories]
            

        image_file_list = []
        for index, directory in enumerate(self.directory_storage.directories):
            tmp_list = []
            tmp_list.extend(self.__retrieve_files__(directory, self.file_type))
            if self.directory_storage.save_dir != None:
                current_save = self.directory_storage.save_dir[index]
                self.directory_storage.save_dir_dict.update({file: current_save for file in tmp_list})
            image_file_list.extend(tmp_list)
        return image_file_list
    
    def __create_mask(self, image: np.ndarray) -> Union[np.ndarray, None]:
        """
        Run the segmentation algorithm on the image.
        Checks first if the image is empty.
        """
        
        try:
            if image_functions.is_image_empty(image, pixels_to_count=10000, background_threshold=3):
                return None
            
            else:
                mask = self.segmentation_class.generate_mask(image)
                return mask.astype(np.uint8) if mask is not None else None 
        except Exception as e:
            raise ValueError(f'{e}')
            
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
    
    def plot_image(self, index, time_series_plot = False, figsize=False, get_mask=False):
        """
        Stub method for plotting an image.
        """
        image_to_plot = self.directory_storage.total_images[index]
        image_array = self.__open_image(image_to_plot)

        col = len(image_array)
        if type(time_series_plot) == int:
            time_index = image_array.isel(t=time_series_plot)
            row=1
        if not time_series_plot:
            time_index = image_array.isel(t=0)
            row=1
        fig, ax = plt.subplots(row, col, )
        
        plt.show()
    
    def get_image(self, image) -> str:
        """
        Open the image at the specified index and plot it.
        """
        return None
        
    def __open_image(self, image_file) -> xarray.DataArray:  # Replace Any with the appropriate type, e.g. xarray.DataArray
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


    def __append_df(self):
        """
        Append dataframe information.
        """
        pass

    def save_masks(self, mask_dict):
        
        
        return None
        
    
    def get_DataFrame(self) -> pd.DataFrame:
        """
        Return the current DataFrame.
        """
        return self.dataframe_storage.df

    def set_channel_names(self, channel):
        """
        Set channel names (to be implemented).
        """
        pass

    def duplicate_image(self, image):
        """
        Duplicates the input image by copying it to a temporary working directory.
        
        Parameters:
        - image (str): Full path to the source image file.
        
        Returns:
        - str: Full path to the duplicated temporary image, or None if copying fails.
        """
        # Get the current working directory
        current_path = os.getcwd()

        # Define the path to the temporary directory where the image will be copied
        tmp_duplicate_dir = os.path.join(current_path, 'img_working_dir')

        # Create the temporary directory if it doesn't already exist
        if not os.path.exists(tmp_duplicate_dir):
            os.mkdir(tmp_duplicate_dir)

        # Define the name of the temporary working image
        working_image = 'tmp_working_image.tif'
        tmp_working_image = os.path.join(tmp_duplicate_dir, working_image)

        try:
            # Copy the original image to the temporary location
            print('Copying image to temporary directory...')
            shutil.copy(image, tmp_working_image)
            print(f"Image copied to {tmp_duplicate_dir} as {working_image}")
        except Exception as e:
            # Handle any errors that occur during copying
            print(f"Error copying image: {e}")
            return None

        # Return the full path to the duplicated image
        return tmp_working_image

    def remove_tmp_image(self, tmp_image):
        """
        Removes a temporary image file if it exists.
        
        Parameters:
        - tmp_image (str): Full path to the temporary image file to remove.
        """
        try:
            # Check if the file exists before attempting to remove it
            if os.path.exists(tmp_image):
                os.remove(tmp_image)
                print("Temporary file removed.")
            else:
                print("Temporary file not found.")
        except Exception as e:
            # Handle any errors that occur during file deletion
            print(f"Error removing file: {e}")
        
    def segment_image(self, image, process_all: bool = True) -> pd.DataFrame:
        """
        Process segmentation for a single image.
        Errors are caught and can be tracked.
        """
        #check image length, if too long, duplicate and open that image, then delete it. 
        if len(image) > 200:
            new_image = self.duplicate_image(image)
            tmp_image = self.__open_image(new_image)
            self.remove_tmp_image(new_image)
        else:
            tmp_image = self.__open_image(image)
        
        self.image_data_storage.image_size_dict = {key: tmp_image[key] for key in tmp_image.sizes}
        self.image_data_storage.image_dict = segmentation_functions.flatten_time_series(tmp_image)      
        if self.image_data_storage.construct == self.acceptor:
            
            seg_ch_images = segmentation_functions.get_segmentation_channel(self.image_data_storage.image_dict, 1)
        else:
            
            seg_ch_images = segmentation_functions.get_segmentation_channel(self.image_data_storage.image_dict, self.segmentation_channel)
        st = datetime.now()    
        
        if self.experiment == 'osmotic':
            seg_ch_images = seg_ch_image[0]
        
        self.image_data_storage.masks = self.__create_masks(seg_ch_images)
        en = datetime.now()
        print((en-st).total_seconds())
        filtered_mask = self.__filter_masks(self.image_data_storage.masks) if self.image_data_storage.masks is not None else None
        
        if self.plot_masks:
            # Code to plot the mask (if implemented)
            pass
        
        if self.save_mask:
            # Code to save the mask (if implemented)
            pass
        
        image_properties = segmentation_functions.get_image_data(filtered_mask, self.image_data_storage.image_dict, self.properties, self.experiment)
        df = segmentation_functions.convert_dict_to_dataframe(image_properties)
    
        return df
    
    def image_segmentation(self) -> pd.DataFrame:
        """
        Perform segmentation on all images and return a consolidated DataFrame.
        """
        # Assume image_list is derived from total_images.

        start_time = datetime.now()
        for image in self.directory_storage.total_images:
            self.tracker.display_progress()    
            self.image_data_storage.image_file = image
            self.image_data_storage.image_name = self.image_data_storage.get_image_name()
            self.image_data_storage.data_columns = self.image_data_storage.get_image_column_variables()
            self.experiment = generic_handlers.get_experiment(self.image_data_storage.data_columns)
            self.image_data_storage.construct = self.image_data_storage.get_construct()
            # Optionally set construct from data_columns, e.g.:
            # self.image_data_storage.construct = self.image_data_storage.data_columns.get('construct', None)
            self.successfully_processed=False
            
            df = self.segment_image(image)
            if not df.empty:
                
                for key, value in self.image_data_storage.data_columns.items():
                    df[key] = value
                self.dataframe_storage.concat_data(df)
                self.successfully_processed=True
            if self.successfully_processed:
                self.tracker.remove_file_from_list(self.image_data_storage.image_file)
            elif not self.successfully_processed:
                try:
                    self.tracker.file_failed(self.image_data_storage.image_file)

                except:
                    self.tracker.tracker_data['failed_files'].append(self.image_data_storage.image_file)
            self.tracker.display_progress()
            end_time = datetime.now()
            #self.elapsed_time += (end_time - start_time).total_seconds()
            '''
            except Exception as e:
                self.tracker.remove_file_from_list(self.image_data_storage.image_file)
                self.tracker.display_progress()
                self.tracker.save_tracker()
            '''
            try:
                clear_output(wait=True)
            except ImportError:
                pass
            
        if len(self.tracker.get_failed_files()) == 0:
            self.tracker.reset_tracker()
        else: 
            self.tracker.get_failed_files()
            
        self.tracker.display_progress()
                
        return self.dataframe_storage.df

    def save_dataframe(save_location:str, file_name: str):
        try:
            self.dataframe_storage.df.to_csv(save_location + filename)
            print(f"Dataframe successfully save to {save_location} as {file_name}")
        except:
            print(f"Dataframe was not saved {save_location}")