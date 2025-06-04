# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:52:32 2025

@author: patri
"""

from typing import Dict, List, Callable, Any
import xarray
import pandas as pd
import numpy as np
import skimage as ski


def flatten_time_series(image: xarray.DataArray) -> dict[xarray.DataArray]:
    '''
    creates a diction to loop over to get the first channel (or selected channel) for segmentation
    runs before get channel and get_segmentation
    '''
    if 't' in image.dims:
        tmp_dic = {timepoint: image.isel(t=timepoint) for timepoint in range(image.sizes['t'])}
    else:
        tmp_dic = {0: image}
    return tmp_dic


def get_channel(image: xarray.DataArray, channel) -> np.array:
    '''
    Gets the selected channel from the xarray, this channel is most likely going to be turned into a mask
    '''
    return image.isel(ch=channel).values


def get_segmentation_channel(image_dic: Dict[int, xarray.DataArray], channel) -> Dict[int, np.array]:
    '''
    Returns a dictionary of 2d xarrays from the channel for segemntation. 
    Example is channel 0 is almost always the donor, so want to select that channel, the send it through segmentation.
    '''
    segmentation_dic= {}
    for key, value in image_dic.items():
        #should be a dictionary of timepoints with segmentation, if osmotic perturbation, then this still works.
        segmentation_dic[key] = get_channel(value, channel)
    
    return segmentation_dic


def get_current_frame(image_dict, index):
    '''
    reutruns the image at the given key index
    '''
    return image_dict[index]


def check_array_type(mask) -> bool:
    """
    Return True if mask is a numpy ndarray.
    """
    return type(mask) == np.ndarray


def check_dtype(mask) -> bool:
    """
    Return True if mask is of type np.uint8.
    """
    return mask.dtype == np.uint8


def check_mask(mask) -> bool:
    """
    Check if mask is a numpy array of type np.uint8.
    """
    if not check_array_type(mask):
        return False
    if not check_dtype(mask):
        return False
    return True


def convert_mask(mask) -> np.array:
    '''
    checks mask to be convert to a uint8 numpy array
    '''
    
    if check_array_type(mask):
        mask= mask.values
    if check_dtype(mask):
        mask = mask.astype(np.uint8)
    return mask

def osmotic_perturbation(mask_dict: Dict[int, np.array or xarray.DataArray],
                   image_dict: Dict[int, xarray.DataArray],
                   properties: List[str])-> Dict[int, Dict[str, np.array]]:

    
    mask = mask_dict[0]
    if type(current_mask) == None: 
        return None
        
    if not check_mask(current_mask):
        current_mask = convert_mask(current_mask)

    properties_dict = {}
    for key, dict in image_dict.item():
        current_image = get_current_frame(image_dict, key)

        channel_dict = {}
        # Loop over each channel in the current image.
        for channel in range(current_image.sizes['ch']):
            # Extract the image data for the given channel.
            image_channel = get_channel(current_image, channel)
            # Compute region properties for the current channel using the mask and image channel.
            try:
                props = ski.measure.regionprops_table(
                    mask, 
                    intensity_image=image_channel, 
                    properties=properties
                )
            except Exception as e:
                raise ValueError(f'{e}')
            channel_dict['ch-' + str(channel)] = props
        # Assign the channel data for the frame to the main dictionary using a key like 'Frame-<key>'.
        properties_dict['Frame-' + str(key)] = channel_dict
    return properties_dict
    



def nonosmotic_perturbation(mask_dict: Dict[int, np.array or xarray.DataArray],
                   image_dict: Dict[int, xarray.DataArray],
                   properties: List[str])-> Dict[int, Dict[str, np.array]]:
    """
    Returns the property values of a whole image (which may have timepoints or not)
    for each frame in mask_dict using the corresponding image from image_dict.
    
    Parameters
    ----------
    mask_dict : dict[int, np.ndarray or xr.DataArray]
        Dictionary mapping frame indices to masks.
    image_dict : dict[int, xr.DataArray]
        Dictionary mapping frame indices to images.
    properties : list of str
        List of properties to extract via skimage.measure.regionprops_table.
    
    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        Nested dictionary with frame keys mapping to channel dictionaries.
    """
    # Initialize an empty dictionary to store properties for each frame.
    properties_dict = {}

    # Iterate over each key in the mask_dict.
    for key, value in mask_dict.items():
        # Get the corresponding image for the current frame.
        current_image = get_current_frame(image_dict, key) 
        # Get the mask for the current frame.
        current_mask = mask_dict[key]
        # Check if the mask is valid; if not, convert it.
        
        
        if type(current_mask) == None: 
            continue
        
        if not check_mask(current_mask):
            current_mask = convert_mask(current_mask)
        # Initialize a dictionary to store data for each channel.
    
        channel_dict = {}
        # Loop over each channel in the current image.
        for channel in range(current_image.sizes['ch']):
            # Extract the image data for the given channel.
            image_channel = get_channel(current_image, channel)
            # Compute region properties for the current channel using the mask and image channel.
            try:
                props = ski.measure.regionprops_table(
                    current_mask, 
                    intensity_image=image_channel, 
                    properties=properties
                )
            except Exception as e:
                raise ValueError(f'{e}')
            # Store the computed properties in the channel_dict with a key like 'ch-0', 'ch-1', etc.
            channel_dict['ch-' + str(channel)] = props
        # Assign the channel data for the frame to the main dictionary using a key like 'Frame-<key>'.
        properties_dict['Frame-' + str(key)] = channel_dict
    return properties_dict

#fix this so channel dicts can be fixed
def set_channel_dict(channel_dict):
    
    if channel_dict == None:
        channel_names = {
            'ch-0' : 'donor', 
            'ch-1' : 'acceptor',
            'ch-2' : 'directAcceptor',
            'ch-3' : 'mCherry',
        }   
    else:
        channel_names = channel_dict
    return channel_names

def get_image_data(mask_dict: Dict[int, np.array or xarray.DataArray],
                   image_dict: Dict[int, xarray.DataArray],
                   properties: List[str], 
                   experiment) -> Dict[int, Dict[str, np.array]]:
    if experiment == 'osmotic':
        return osmotic_perturbation(mask_dict, image_dict, properties)
    else:
        return nonosmotic_perturbation(mask_dict, image_dict, properties)

#fix this so channel dicts can be fixed
def get_channel_name(channel_index, channel_names=None, experiment_type=None):
    """
    Returns a human-readable channel name given a channel index and (optionally) an experiment type.

    Parameters
    ----------
    channel_index : str or int
        The channel key/index (e.g., 'ch-0', 'ch-1').
    channel_dict: Dict that gives, user the option to pass a dictionary higher up for the names of each channel, otherwise sets a default list
    experiment_type : str, optional
        The type of experiment. If set to 'osmotic_perturbation', custom logic may be applied.

    Returns
    -------
    str
        The channel name.
    
    Raises
    ------
    KeyError
        If the channel_index is not found in the channel dictionary.


        design to give the flexibility of adding additional dictionaries based on experimental need.
    """
    channel_dict = set_channel_dict(channel_names)
    
    if experiment_type == 'osmotic_perturbation':
        # TODO: implement osmotic_perturbation logic
        raise NotImplementedError("osmotic_perturbation channel mapping not implemented.")
    try:
        # Ensure that the channel key is a string (if not, convert it).
        key = channel_index if type(channel_index)==str else f'ch-{channel_index}'
        return channel_dict[key]
    except KeyError:
        raise KeyError(f"Channel key '{channel_index}' not found in channel mapping.")


#fix this so people can pass additional properties to the list that will be removed
def check_column_name(column):
    """
    Checks whether a given column name should remain unchanged (i.e. not prefixed).

    Parameters
    ----------
    column : str
        The column name to check.

    Returns
    -------
    bool
        True if the column is in the list of columns that should remain unchanged,
        otherwise False.
    """
    # Define list of columns that should not be prefixed.
    column_list = [
        'label',
        'area',
        'centroid',
        'solidity', 
        'num_pixels', 
        'orientation',
    ]
    return column in column_list


def convert_dict_to_dataframe(properties_dict: Dict, merge='inner', on='label', experiment=None) -> pd.DataFrame:
    """
    Converts a nested dictionary of properties into a single pandas DataFrame.

    The expected structure is:
        properties_dict = {
            'Frame-<frame>': {
                'ch-0': <data for channel 0>,
                'ch-1': <data for channel 1>,
                ...
            },
            ...
        }
    For each frame key, the inner dictionary is converted into a DataFrame.
    Channel names are mapped via get_channel_name (and prefixed to column names except
    for keys such as 'label' and 'area').
    The DataFrames for different channels are merged (using the 'on' column),
    and a 'timepoint' column is added based on the frame extracted from the key.

    Parameters
    ----------
    properties_dict : dict
        A nested dictionary containing image property data for each frame and channel.
    merge : str, optional
        Merge method for DataFrame.merge, default is 'inner'.
    on : str, optional
        Column name on which to merge, default is 'label'.
    experiment : str, optional
        Experiment type (passed to get_channel_name) for custom channel mapping.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame with all frames and their respective properties.

    Raises
    ------
    ValueError
        If a frame key does not contain a '-' to extract the timepoint.
    """
    # Initialize an empty DataFrame to accumulate all frames.
    test_df = pd.DataFrame()
    
    for frame_key, channel_data_dict in properties_dict.items():
        # Expect frame_key to be something like 'Frame-<value>'

        try:
            # Split the key on '-' and take the second part as the timepoint/frame identifier.
            frame = frame_key.split('-')[1]
        except IndexError:
            raise ValueError(f"Frame key '{frame_key}' is not formatted as expected (e.g., 'Frame-<value>').")

        # Initialize an empty DataFrame for the current frame.
        frame_df = pd.DataFrame()
        
        for channel_key, data in channel_data_dict.items():
            # Get the channel name based on the channel key and experiment type.
            
            channel = get_channel_name(channel_key, experiment)
            # Convert the data (expected to be dict-like or array-like) into a DataFrame.
            data_df = pd.DataFrame(data)
            # Get the current columns of this DataFrame.
            columns = data_df.columns
            # Rename columns: if column name should be modified, prefix with channel name.
            new_column_names = [
                f"{channel}_{column}" if not check_column_name(column) else column
                for column in columns
            ]
            data_df.columns = new_column_names

            # If this is the first channel (assumed to be 'ch-0'),
            # simply assign the DataFrame.
            if channel_key == 'ch-0':
                frame_df = data_df.copy()
            else:
                # Otherwise, merge the new channel's DataFrame with the accumulated one.
                try:
                    frame_df = frame_df.merge(data_df, how=merge, on=on, suffixes=('', '_y'))
                    # Remove any columns that ended with '_y' (to avoid duplicate columns).
                    frame_df.drop(frame_df.filter(regex='_y$').columns, axis=1, inplace=True)
                except KeyError as e:
                    raise KeyError(f"Error merging channel data on key '{on}': {e}")
            
        frame_df['timepoint']= frame
        test_df = pd.concat([test_df, frame_df], ignore_index=True)
        
    return test_df