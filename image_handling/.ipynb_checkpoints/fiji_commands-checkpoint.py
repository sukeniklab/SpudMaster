import xarray
import image_handling.image_functions as image_functions

def read_images_to_python(ij, image: str) -> xarray.DataArray:
    """
    Reads an image using ImageJ and converts it into an xarray.DataArray.

    Parameters:
        ij: The ImageJ gateway instance used to handle image I/O.
        image (str): The file path to the image to be read.

    Returns:
        xarray.DataArray: The image data as an xarray.DataArray.
    
    Raises:
        FileNotFoundError: If the specified image file cannot be found.
        ValueError: If the image cannot be converted to an xarray.DataArray.
    """
    try:
        # Open the image using ImageJ
        tmp = ij.io().open(image)
        
        # Convert the image to Python format
        img = ij.py.from_java(tmp)

        #close image
        
        # Ensure the result is an xarray.DataArray
        if not isinstance(img, xarray.DataArray):
            raise ValueError(f"The loaded image could not be converted to xarray.DataArray. Got type: {type(img)}")

        return img

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {image}") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the image '{image}': {str(e)}") from e


def concat_xarrays(ij, image_list: list) -> xarray.DataArray:
    """
    Concatenates a list of images loaded as xarray.DataArray along the 'ch' dimension.

    Parameters:
        ij: An ImageJ instance used for reading the images.
        image_list (list): A list of image file paths to be loaded and concatenated.

    Returns:
        xr.DataArray: The concatenated xarray.DataArray with a consistent 'ch' dimension.
    """
    tmp_xarr = None

    for image_path in image_list:
        # Read the image into an xarray.DataArray
        tmp = read_images_to_python(ij, image_path)

        # Ensure 'ch' dimension exists, and make it the last dimension
        if 'ch' not in tmp.dims:
            tmp = tmp.expand_dims(dim="ch").assign_coords(ch=[2])  # Assign default channel value
            tmp = tmp.transpose(..., "ch")  # Ensure 'ch' is the last dimension

        # Reshape the image (ensure consistent dimensions)
        tmp = image_functions.reshape_image(tmp)

        # Concatenate along 'ch'
        if tmp_xarr is None:
            tmp_xarr = tmp
        else:
            # Ensure proper alignment of 'ch' coordinates
            tmp_xarr = xr.concat([tmp_xarr, tmp], dim="ch")

    return tmp_xarr