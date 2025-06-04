import os
import gc
import xarray as xr
import numpy as np
from skimage import io
from typing import List, Dict, Callable
from key_class import Key
from key_handlers import key_functions
from file_handling import generic_handlers  
from image_handling import image_functions
from image_handling import fiji_commands
from Tracker import Tracker  # Assuming Tracker is defined in Tracker.py
from datetime import datetime
from IPython.display import clear_output
import time

class ImageAlignmentAndTifConversion:
    """
    This class handles the alignment of images and conversion to TIFF format.
    
    It integrates:
      - A Key instance for preloaded save filename information.
      - An ImageJBridge instance for executing Fiji/ImageJ commands.
      - A Tracker instance to monitor progress, handle errors, and resume processing.
      - Directory management for saving aligned images.
    
    The class provides a high-level process_images() function that scans directories,
    loads file paths into the tracker, and for each set of images (grouped by well and replicate)
    performs image alignment (via Fiji commands and custom functions) before saving a TIFF.
    """
    def __init__(self, 
                 ij: Callable, 
                 observation_methods: int = 1, 
                 initialized_key: Key = None,
                 tracker_name: str = 'tracking_file.json'):
        
        
        try:
            self.ij = ij # Use the ImageJ instance from the bridge
        except Exception as e:
            raise ValueError("Failed to initialize ImageJ instance.") from e
        
        # Initialize tracker (handles progress and errors)
        self.file_tracker = Tracker(tracker_name)
        self.cam2_alignment_register = None
        # The key holds pre-built save file naming information.
        if initialized_key is None:
            raise ValueError("A Key instance must be provided for save file information.")
        self.key = initialized_key
        
        self.file_list = []
        # Obtain observation method details from generic handlers.
        try:
            self.obs_method = generic_handlers.SetObservationMethod(obs_method_number=observation_methods).get_observation_method()
        except Exception as e:
            raise RuntimeError("Failed to get observation method from generic handlers.") from e
        
        self.current_image_set = []
        self.pixel_count = None
        self.background_threshold = None
        self.__new_image_dir__ = None

    def __retrieve_files__(self, directory: str, file_type: str) -> List[str]:
        """
        Retrieve all files with the given extension from a directory.
        """
        try:
            file_list = generic_handlers.get_files_by_extension(directory, file_type)
            return file_list
        except Exception as e:
            raise RuntimeError(f"Error retrieving files from {directory}") from e

    def __concat_aligned_images__(self, input_image: xr.DataArray) -> xr.DataArray:
        """
        Placeholder for a function to concatenate aligned images.
        Extend this function as needed.
        """
        # For now, simply return the input image.
        return input_image

    def __create_new_directory__(self, base_directory: str) -> str:
        """
        Create (if needed) and return a new directory for saving aligned images.
        """
        new_dir = os.path.join(base_directory, "aligned_images")
        try:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            self.__new_image_dir__ = new_dir
            return new_dir
        except Exception as e:
            raise RuntimeError("Failed to create new directory for aligned images.") from e

    def _save_to_tif(self, directory: str, image: xr.DataArray, filename: str):
       
        # If new image directory not set, create it.
        if not os.path.isdir(directory):
            self.__new_image_dir__ = directory
            save_path = os.path.join(self.__new_image_dir__, filename)
        else:
            try:
                self.__new_image_dir__ = self.__create_new_directory__(directory)
                save_path = os.path.join(self.__new_image_dir__, filename)
            except Exception as e:
                raise ValueError("New image directory not set and failed to create one.") from e
        
        try:
            save_image = image.transpose('t', 'row', 'col', 'ch')
        except:
            save_image = image.transpose('row', 'col', 'ch')
        
        save_image.sizes
        try: 
            ij_img = self.ij.py.to_java(save_image)
            ij_img = self.ij.py.to_imageplus(ij_img)
            self.ij.IJ.save(ij_img, save_path)
            ij_img.close()
        except Exception as e:
            raise RuntimeError(f"Failed to save TIFF file to {save_path}") from e

    def _image_aligning(self, 
                        image_set: List[str], 
                        stat_image: int, 
                        alignment_method, 
                        excluded_channel: List[int], 
                        stack_method, 
                        save_directory: str,
                        well: str):
        """
        Align a set of images and save the final TIFF file.
        
        Parameters:
            image_set: List of file paths for images to align.
            stat_image: Index of the stationary image.
            alignment_method: Alignment method to use.
            excluded_channel: List of channels to exclude.
            stack_method: Method for stacking images.
            save_directory: Directory to save the aligned image.
            well: Well identifier for validation and naming.
        """
        
        try:
            # Concatenate the input images using Fiji commands.
            images = fiji_commands.concat_xarrays(self.ij, image_set)
            
        except Exception as e:
            raise ValueError("Error opening and concatenating files via Fiji commands.") from e

        # Use the first image filename as template for final filename.
        image_filename = image_set[0]
        well_from_filename = generic_handlers.get_well_from_file(image_filename)
        try:
            temp_wellname = well_from_filename.split('-')[0]
            if well !=  temp_wellname:
                raise ValueError(f"Mismatch: key well '{well}' vs. image well '{well_from_filename}'.")
        except: 
            if well_from_filename != well:
                raise ValueError(f"Mismatch: key well '{well}' vs. image well '{well_from_filename}'.")
        # Retrieve pre-built file save name from the key.
        tif_save_filename = self.key.current_filename_save_parameters(image_filename, well)
        # Get construct information from key (e.g., a column like 'construct')
        try:
            current_construct = self.key.get_col_by_well(col='construct', well=well)
            
        except Exception as e:
            raise RuntimeError("Error retrieving construct information from Key.") from e
        
        # Adjust parameters based on construct.
        if current_construct.iloc[0] == 'mTQ2':
            excluded_channel.append(2)  # Append channel index 2
        if current_construct.iloc[0] == 'mNG':
            stat_image = 1  # Use second image as stationary
        
        # Align images: if a time dimension exists, process each timepoint.
        try:
            if 't' in images.dims:
                concat_list = []
                start = datetime.now()
                for timepoint in range(images.sizes['t']):
                    cur_image_t = images.isel(t=timepoint)
                    if self.cam2_alignment_register == None:
                        try:
                            self.cam2_alignment_register = image_functions.register_alignment(cur_image_t.isel(ch=0), cur_image_t.isel(ch=1),  stack_method)
                            print(self.cam2_alignment_register)
                        except Exception as E:
                            raise RuntimeError(f"Error aligning.{e}") from e
                    aligned = image_functions.image_alignment(cur_image_t, stat_image, excluded_channel, self.cam2_alignment_register)
                    concat_list.append(aligned)
                aligned_image = xr.concat(concat_list, dim='t')
                end = datetime.now()
                print((end - start).total_seconds())
            else:
                
                aligned_image = image_functions.image_alignment(images, stat_image, excluded_channel, stack_method)
            
        except Exception as e:
            raise RuntimeError("Error during image alignment process.") from e
        
        # Save the aligned image.
        try:
            self._save_to_tif(save_directory, aligned_image, tif_save_filename)
        except Exception as e:
            # Log error in tracker for each file in image_set.
            for file in image_set:
                self.file_tracker.evaluate_error(file, e)
            # Optionally, move files to an error directory or remove them.
            for file in image_set:
                self.file_tracker.remove_file_from_list(file)
            # Re-raise the error to indicate failure.
            raise

        # On success: remove processed files from tracker and display progress.
        for file in image_set:
            self.file_tracker.remove_file_from_list(file)
        

    def process_images(self, 
                       directories: List[str], 
                       stationary_image: int = 0,
                       excluded_channels: List[int] = [], 
                       stack_method = None,
                       file_type: str = 'vsi',
                       save_dir: str = None):
        """
        Process images from one or more directories:
          - Retrieve files matching the given extension.
          - Load file paths into the tracker (or resume from previous state).
          - Group files by well and replicate.
          - For each group, call _image_aligning to align images and save the result.
          - Reset the tracker at the end.
        
        Parameters:
            directories: List of directory paths to search.
            stationary_image: Index for the stationary image.
            excluded_channels: List of channels to exclude.
            stack_method: The stacking method; if not provided, defaults to 'rigid_body'.
            file_type: File extension to search for (e.g., 'vsi').
            save_dir: Directory in which to save aligned images.
        """
        try:
            # Ensure directories is a list.
            
    
            # Gather image files from all directories.
            image_file_list = []
            if self.file_tracker.current_file_list():
                image_file_list.extend(self.file_tracker.current_file_list())
            else:
                if not isinstance(directories, list):
                    directories = [directories]
                for directory in directories:
                    image_file_list.extend(self.__retrieve_files__(directory, file_type))
            

            # Load files into the tracker if not already loaded.
            if not self.file_tracker.current_file_list():
                self.file_tracker.load_files(image_file_list)
            self.file_tracker.display_progress()
    
            # Get unique wells from the file list.
            experimental_wells = generic_handlers.get_unique_wells_in_experiment(image_file_list)
            # Build a dictionary mapping each well to its list of files.
            image_dic = {well: key_functions.get_unique_well_list(image_file_list, well) for well in experimental_wells}
            
            #image_dic[well] = [f for f in image_file_list if generic_handlers.get_well_from_file(f) == well]
            
            # Optionally, build replicate information for each well.
            replicate_dic = {}
            for well, files in image_dic.items():
                
                replicate_dic[well] = generic_handlers.loop_iterator(files, self.obs_method)
            
            # For each well and replicate, process alignment.
            for well, files in image_dic.items():
                # For each replicate (here using loop_iterator to partition files)
                num_replicates = replicate_dic.get(well, 1)
                for rep in range(num_replicates+1):
                    try:
                        # Extract the subset of files for this replicate.
                        # (Assumes extract_experiment_images handles the replicate extraction.)
                        replicate_files = generic_handlers.extract_experiment_images(files, rep)
                        if not replicate_files:
                            continue
                        # Ensure a save directory exists.
                        if save_dir is None:
                            base_dir = os.path.dirname(replicate_files[0])
                            save_directory = self.__create_new_directory__(base_dir)
                        else:
                            save_directory = save_dir
                            
                        # Align images for this replicate.
                        
                        replicate_files = generic_handlers.sort_filenames(replicate_files, self.obs_method)
                        self.file_tracker.display_progress()
                        self._image_aligning(replicate_files, stationary_image, 
                                             alignment_method=None,  # Pass alignment_method if needed.
                                             excluded_channel=excluded_channels.copy(), 
                                             stack_method=stack_method if stack_method is not None else image_functions.stack_method('rigid_body'),
                                             save_directory=save_directory,
                                             well=well)
                        time.sleep(1.5)
                        clear_output(wait=False)
                        gc.collect()
                    except Exception as rep_e:
                        print(f"Error aligning images for well {well}, replicate {rep}: {rep_e}")
                        # Continue with next replicate.
                        continue
            
            # Once done, reset tracker.
            self.file_tracker.reset_tracker()
    
        except Exception as e:
            raise Exception("Could not complete processing images.") from e