# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:07 2025

@author: patri
"""

import os 
from typing import List

class SetObservationMethod:
    def __init__(self, obs_method_number: int = 1, custom_observation: list[str] = None):
        """
        Initialize the SetObservationMethod class.

        Parameters:
            obs_method_number (int): The observation method number to use (1-5).
            custom_observation (list[str], optional): A custom list of observation methods.
        """
        self.obs_method_number = obs_method_number
        self.custom_observation = custom_observation
        self.obs_method = None

        # Validate custom_observation if provided
        if self.custom_observation is not None:
            if not isinstance(self.custom_observation, list):
                raise ValueError(
                    f"Expected custom_observation to be of type list, got {type(self.custom_observation)}"
                )

        # Define observation methods
        self.observation_methods = {
            1: ["FRET", "DirectA"],
            2: ["FRET", "DirectA", "mCherry"],
            3: ["FRET", "DirectA", "DAPI"],
            4: ["FRET", "DirectA", "mCherry", "BF"],
            5: ["FRET", "DirectA", "DAPI", "BF"],
        }

    def __selected_method__(self) -> list[str]:
        """
        Select the observation method based on the input parameters.

        Returns:
            list[str]: The selected observation method.
        """
        if self.custom_observation is not None:
            self.obs_method = self.custom_observation
        else:
            if self.obs_method_number not in self.observation_methods:
                raise ValueError(
                    f"Invalid obs_method_number {self.obs_method_number}. Valid options are {list(self.observation_methods.keys())}"
                )
            self.obs_method = self.observation_methods[self.obs_method_number]
        return self.obs_method

    def get_observation_method(self) -> list[str]:
        """
        Public method to get the selected observation method.

        Returns:
            list[str]: The selected observation method.
        """
        return self.__selected_method__()

def get_file_len(filename, separator):
    return len(os.path.basename(filename).split('.')[0].split('_'))

def get_first_file(file_list, file_type):
    for filename in file_list:
        if filename.endswith(file_type):
            return filename
    return None

def dir_file_split_len(directory, file_type='vsi', separator='_'):
    file_list = os.listdir(directory)
    file = get_first_file(file_list, file_type)
    if file is None:
        raise f'file of type {file_type} was not found.'
    return get_file_len(file, separator)
        
def get_well_from_file(filename:str)-> str:
    return os.path.basename(filename).split('.')[0].split('_')[-1]

def sort_filenames(filenames: List[str], sort_order: List[str]) -> List[str]:
    """
    Sorts a list of filenames based on the first occurrence of keywords provided in sort_order.
    
    Parameters:
        filenames (List[str]): List of filenames to be sorted.
        sort_order (List[str]): List of keywords defining the desired sort order.
        
    Returns:
        List[str]: Sorted list of filenames.
    """
    def sort_key(filename: str) -> int:
        for index, keyword in enumerate(sort_order):
            if keyword in filename:
                return index
        return len(sort_order)  # Filenames with no keyword will be placed last.
    
    return sorted(filenames, key=sort_key)

def get_files_by_extension(directory: str, extension: str):
    """
    Get a list of all files with a given extension from a directory.

    :param directory: Path to the directory to search in.
    :param extension: File extension (e.g., ".txt", ".jpg").
    :return: List of matching file paths.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")
    
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]



def split_filename(image: str, separator: str) -> list:
    """
    Splits a filename into parts using a given separator.

    :param image: Full file path or filename.
    :param separator: Character(s) used to split the filename.
    :return: List of extracted parts.
    """
    if not isinstance(image, str) or not image:
        raise ValueError("Invalid filename provided.")

    # Extract just the filename without the directory
    filename = os.path.basename(image)
    
    # Remove file extension, if present
    name_only = os.path.splitext(filename)[0]

    # Split using the provided separator
    return name_only.split(separator)

def get_experiment(file_dict: dict):
    return file_dict['experiment'].lower()
    
def get_unique_wells_in_experiment(experiment_list: list) -> list:
    """
    Extracts and returns a sorted list of unique wells from the given experiment list.

    Parameters:
        experiment_list (list): A list of strings representing experiment filenames.

    Returns:
        list: A sorted list of unique wells in the experiment.
    """
    # Extract the wells from the filenames
    wells_in_experiment = [split_filename(x, '_')[-1] for x in experiment_list]

    # Check if any wells contain replicates (indicated by '-')
    if any("-" in item for item in wells_in_experiment):
        # Split at '-' and take the well ID (ignoring replicates)
        wells_in_experiment = [well_replicate.split("-")[0] for well_replicate in wells_in_experiment]

    # Remove duplicates and sort the wells
    wells_in_experiment = sorted(set(wells_in_experiment))

    return wells_in_experiment


def loop_iterator(image_list: list, unique_observation_methods: list) -> int:
    """
    Calculates the number of iterations needed to process all images 
    using the given unique observation methods.
    
    Parameters:
    image_list (list): A list containing image data.
    unique_observation_methods (list): A list containing unique observation methods.
    
    Returns:
    int: The number of iterations needed.
    
    Raises:
    ValueError: If either list is empty to prevent division by zero.
    TypeError: If inputs are not lists.
    """
    
    # Validate input types
    if not isinstance(image_list, list) or not isinstance(unique_observation_methods, list):
        raise TypeError("Both image_list and unique_observation_methods must be of type list.")
    
    # Validate input content
    if not image_list:
        raise ValueError("image_list cannot be empty.")
    if not unique_observation_methods:
        raise ValueError("unique_observation_methods cannot be empty.")
    
    # Compute the number of iterations needed (integer division for whole iterations)
    return len(image_list) // len(unique_observation_methods)



def extract_experiment_images(image_list: list, unique_replicate: int) -> list:
    """
    Extracts images from the image list that match a specific unique replicate identifier.
    
    Parameters:
    image_list (list): A list of image filenames.
    unique_replicate (int): A unique replicate identifier to filter images.
    
    Returns:
    list: A list of filtered image filenames that match the unique replicate identifier.
    
    Raises:
    ValueError: If image_list is empty.
    TypeError: If image_list is not a list or unique_replicate is not an integer.
    """
    
    # Validate input types
    if not isinstance(image_list, list):
        raise TypeError("image_list must be of type list.")
    if not isinstance(unique_replicate, int):
        raise TypeError("unique_replicate must be an integer.")
    
    # Validate input content
    if not image_list:
        raise ValueError("image_list cannot be empty.")
    
    # Extract images matching the unique replicate identifier
    return [x for x in image_list if '-' + str(unique_replicate) in x.split("_")[-1]]



''' 
def csv_save():
    
def image_save():
    
def 
'''