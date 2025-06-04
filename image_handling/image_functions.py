import xarray
import numpy as np
from pystackreg import StackReg as sr

def is_image_empty(img: xarray.DataArray or np.array, pixels_to_count: int = 10000, background_threshold: float = 3.0) -> bool:
    """
    Determine if an image is "empty" by analyzing the intensity difference between 
    the brightest and darkest pixels.

    Parameters:
        img (xr.DataArray): The input image as an xarray.DataArray.
        pixels_to_count (int): The number of brightest and darkest pixels to analyze.
        background_threshold (float): The threshold ratio to determine emptiness.
                                      If the ratio of average brightest to darkest pixels 
                                      is below this value, the image is considered empty.

    Returns:
        bool: True if the image is considered empty, False otherwise.

    Raises:
        ValueError: If the image has fewer pixels than `pixels_to_count`.
    """
    # Convert the xarray to a NumPy array
    if isinstance(img, xarray.DataArray):
        img = img.values
    
    # Flatten the image to simplify pixel intensity analysis
    flatten_image = np.ravel(img)

    # Validate that there are enough pixels to analyze
    if flatten_image.size < pixels_to_count:
        raise ValueError(
            f"Image has fewer pixels ({flatten_image.size}) than `pixels_to_count` ({pixels_to_count})."
        )

    # Extract the `pixels_to_count` lowest and highest pixels
    lowest_pixels = np.partition(flatten_image, pixels_to_count)[:pixels_to_count]
    highest_pixels = np.partition(flatten_image, -pixels_to_count)[-pixels_to_count:]

    # Compute the mean of the lowest and highest pixel groups
    average_lowest_pixels = lowest_pixels.mean()
    average_highest_pixels = highest_pixels.mean()

    # Calculate the intensity difference ratio
    pixel_difference = average_highest_pixels / average_lowest_pixels

    # Return True if the ratio is below the background threshold
    return pixel_difference < background_threshold




def reshape_image(input_array: xarray.DataArray, new_dim_list=None) -> xarray.DataArray:
    """
    Reshape an xarray.DataArray by transposing its dimensions.
    
    Parameters:
        input_array (xr.DataArray): The input xarray DataArray to be reshaped.
        new_dim_list (list, optional): A list of dimension names specifying the new order.
                                       If None, a default order based on 'row', 'col', 
                                       and the remaining dimensions will be used.

    Returns:
        xr.DataArray: A transposed xarray.DataArray.
    
    Raises:
        ValueError: If the inferred or provided dimension order is invalid.
    """

    
    # Case 1: If new_dim_list is provided, use it for transposition
    if new_dim_list is not None:
        try:
            # Transpose based on the provided dimension list
            new_array = input_array.transpose(*new_dim_list)
        except ValueError as e:
            raise ValueError(
                f"Error in transposing with new_dim_list={new_dim_list}. Check if all dimensions match the input array."
            ) from e

    # Case 2: If no new_dim_list is provided, use default logic to reorder dimensions
    if  input_array.dims[2] == 'ch':
        new_array = input_array
    
    
    else:
        # Extract current dimensions of the input array
        dims = list(input_array.dims)

        # Ensure 'row' and 'col' exist in the dimensions
        trans_list = ['row', 'col']
        for dim in trans_list:
            if dim in dims:
                dims.remove(dim)
            else:
                raise ValueError(f"Missing dimension '{dim}' in the input array dimensions: {input_array.dims}")

        # Reverse the remaining dimensions and append them to 'row' and 'col'
        remaining_dims = list(reversed(dims))
        trans_list.extend(remaining_dims)

        # Ensure the third dimension is 'ch'
        if len(trans_list) < 3 or trans_list[2] != 'ch':
            raise ValueError(
                f"Invalid dimension order: The third dimension must be 'ch'. "
                f"Current inferred order: {trans_list}. Please pass 'new_dim_list' explicitly."
            )

        # Attempt to transpose the array with the determined order
        try:
            new_array = input_array.transpose(*trans_list)
        except ValueError as e:
            raise ValueError(
                f"Failed to transpose using the inferred dimension order {trans_list}. Ensure the array is correctly structured."
            ) from e

    return new_array




def stack_method(method: str) -> sr:
    """
    Selects and returns the appropriate alignment method for image stacking.

    Parameters:
        method (str): The alignment method to use. Must be one of the following:
                      'translation', 'rigid_body', 'scaled_rotation', 'affine', 'bilinear'.

    Returns:
        sr: The alignment method corresponding to the provided method string.

    Raises:
        ValueError: If an invalid method is provided.
    """
    # Convert method to lowercase for case-insensitive matching
    method = method.lower()

    # Define possible methods and their corresponding sr values
    if method == 'translation':
        return sr(sr.TRANSLATION)
    elif method == 'rigid_body':    
        return sr(sr.RIGID_BODY)
    elif method == 'scaled_rotation': 
        return sr(sr.SCALED_ROTATION)
    elif method == 'affine': 
        return sr(sr.AFFINE)
    elif method == 'bilinear': 
        return sr(sr.BILINEAR)
    else:
        raise ValueError(
            f"Invalid method '{method}'. Possible methods are: {list(method_mapping.keys())}"
        )

def image_alignment(image_array: xarray.DataArray, stationary_channel: int, alignment_method, 
                    excluded_channels: list[int] = None
                    ) -> xarray.DataArray:  
    """
    Aligns a multi-channel image using the specified alignment method.

    Parameters:
        image_array (xr.DataArray): Input image array with dimensions ('row', 'col', 'ch').
        stationary_channel (int): The index of the channel to use as the stationary reference for alignment.
        excluded_channels (list[int], optional): A list of channel indices to exclude from alignment.  # Fixed param name
        alignment_method (object, optional): An object with a `.transform()` method for aligning images.  # Clearer description

    Returns:
        xr.DataArray: The aligned image array with the same dimensions as the input.

    Raises:
        ValueError: If the input array does not have exactly three dimensions or the required dimensions.
    """
    
    dimensions = image_array.dims
    if len(dimensions) != 3:
        raise ValueError(f"Expected 3 dimensions, got {len(dimensions)}")  # Fixed undefined variable `dim_length`

    required_dims = ['row', 'col', 'ch']  #Renamed for clarity (was `require_list`)
    if list(dimensions) != required_dims:
        image_array = reshape_image(image_array, new_dim_list=required_dims)  # Assumes reshape_image is custom function

    if stationary_channel < 0 or stationary_channel >= image_array.sizes['ch']:
        raise ValueError(f"Stationary image index {stationary_channel} is out of bounds for the 'ch' dimension.")

    if excluded_channels is None:
        excluded_channels = []  # Safely handles mutable default argument

    stationary_channel = stationary_channel
    realigned_list = []
    for index in range(image_array.sizes['ch']):
        current_img = image_array.isel(ch=index)
        
        if index in excluded_channels or index == stationary_channel:  # Combined two conditionals for clarity
            realigned_list.append(current_img)
            
            continue
        
        if alignment_method is not None:  # Replaced != None with is not None (PEP 8 style)
            
            registered_image = alignment_method.transform(current_img)
            realigned_list.append(xarray.DataArray(registered_image, dims=current_img.dims, coords=current_img.coords))
            
        else:
            realigned_list.append(current_img)  # Removed redundant wrapping with DataArray again
        
    aligned_images = xarray.concat(realigned_list, dim='ch')
    aligned_images = reshape_image(aligned_images, new_dim_list=required_dims)  # Ensures final dims are consistent
    return aligned_images



'''
def image_alignment(image_array: xarray.DataArray, stationary_channel:int, excluded_channels: list[int], 
                    alignment_method) -> xarray.DataArray:
    """
    Aligns a multi-channel image using the specified alignment method.

    Parameters:
        image_array (xr.DataArray): Input image array with dimensions ('row', 'col', 'ch').
        stationary_image (int): The index of the channel to use as the stationary reference for alignment.
        excluded_channels (list[int]): A list of channel indices to exclude from alignment.
        alignment_method (str): The alignment method to use (e.g., 'rigid_body').

    Returns:
        xr.DataArray: The aligned image array with the same dimensions as the input.
    
    Raises:
        ValueError: If the input array does not have exactly three dimensions or the required dimensions.
    """
     # Validate input dimensions
    dimensions= image_array.dims
    if len(dimensions) != 3:
        raise ValueError(f"expected 3 dimensions, got {dim_length}")
    
    require_list = ['row', 'col', 'ch']
    if list(dimensions) != require_list:
        # Reshape the image to ensure dimensions are in the correct order
        image_array = reshape_image(image_array, new_dim_list=require_list)

    if stationary_channel < 0 or stationary_channel >= image_array.sizes['ch']:
        raise ValueError(f"Stationary image index {stationary_channel} is out of bounds for the 'ch' dimension.")
    
    # Extract the stationary image
    stationary_img = image_array.isel(ch=stationary_channel)
    
    # Handle excluded channels
    if excluded_channels is None:
        excluded_channels = []
    realigned_list = []
    for index in range(image_array.sizes['ch']):
        print(index)
        current_img = image_array.isel(ch=index)
        # Exclude specified channels from alignment
        if index in excluded_channels:
            realigned_list.append(current_img)
            continue
            
        # Skip alignment for the reference image    
        if index == stationary_channel: 
            realigned_list.append(current_img)
            continue

        if alignment_method != None:
            registered_image= alignment_method.transform(current_img)
            realigned_list.append(xarray.DataArray(registered_image, dims=current_img.dims, coords=current_img.coords))
            continue
            
        else:
            realigned_list.append(xarray.DataArray(current_img, dims=current_img.dims, coords=current_img.coords))
            # Align the current image with the stationary image

            
    # Concatenate the aligned images along the 'ch' dimension and ensure the shape is maintained as the input array  
    aligned_images = xarray.concat(realigned_list, dim='ch')
    aligned_images = reshape_image(aligned_images, new_dim_list =require_list)

    # Return the aligned images
    return aligned_images


def align_to_osmo(image_set, alignment_method):
    realigned_list= []
    for index in range(image_set.sizes['ch']):
        current_img = image_set.isel(ch=index)
        registered_image= alignment_method.transform(current_img)
        realigned_list.append(xarray.DataArray(registered_image, dims=current_img.dims, coords=current_img.coords))
    aligned_images = xarray.concat(realigned_list, dim='ch')
    aligned_images = reshape_image(aligned_images, new_dim_list =require_list)

    # Return the aligned images
    return aligned_images
'''

def align_to_osmo(image_set, alignment_method):
    """
    Aligns each channel in the image_set using the provided alignment_method.

    Parameters:
        image_set (xr.DataArray): Input array with dimensions ('row', 'col', 'ch').
        alignment_method: An object with a .transform() method that accepts an image and returns an aligned image.

    Returns:
        xr.DataArray: The aligned image array.
    """

    if 'ch' not in image_set.dims:
        raise ValueError("Expected dimension 'ch' in the image_set")  # Ensures input is as expected

    required_dims = ['row', 'col', 'ch']  # Required for reshaping

    realigned_list = []
    for index in range(image_set.sizes['ch']):
        current_img = image_set.isel(ch=index)
        registered_image = alignment_method.transform(current_img)  #Assumes transform returns numpy or array-like
        realigned_list.append(xarray.DataArray(registered_image, dims=current_img.dims, coords=current_img.coords))

    aligned_images = xarray.concat(realigned_list, dim='ch')
    aligned_images = reshape_image(aligned_images, new_dim_list=required_dims)  #Assumes reshape_image handles dimension order

    return aligned_images