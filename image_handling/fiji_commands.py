import xarray
import image_handling.image_functions as image_functions
import jpype
import gc
from pathlib import Path


def memory_in_MB(memory):
    return memory/1024**2
    
def check_jvm_memory():
    # Access Java runtime
    Runtime = jpype.JClass("java.lang.Runtime")
    runtime = Runtime.getRuntime()
    
    # Force Java GC (optional, to clean stale refs)
    runtime.gc()
    runtime.runFinalization()
    # Fetch memory stats
    total_memory = runtime.totalMemory()    # in bytes
    free_memory = runtime.freeMemory()      # in bytes
    used_memory = total_memory - free_memory
    max_memory = runtime.maxMemory()        # in bytes

    memory_list = [total_memory, free_memory, used_memory, max_memory]
    memory_dict = {}
    for index, memory in enumerate(memory_list):
        memory_dict[index] = memory_in_MB(memory)
    
    # Display in megabytes
    print(f"Used Memory: {memory_dict[2]:.2f} MB")
    print(f"Free Memory: {memory_dict[1]:.2f} MB")
    print(f"Total Allocated Memory: {memory_dict[0]:.2f} MB")
    print(f"Max JVM Memory: {memory_dict[3]:.2f} MB")
    return memory_dict

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
    memory_usage = check_jvm_memory()
    ij.IJ.run("Collect Garbage")
    ij.IJ.run("Fresh Start")
    
    try:
        # Open the image using ImageJ
        
        print('Opening image')
        try:
            open_image = True
            try:
                tmp = ij.io().open(image)
            except:
                img_path = Path(image)
                tmp = ij.io().open(str(img_path))
                
        except:
            print('Warning: Memory will not be cleared properly. If running multiple iterations, you will need to restart.')
            open_image = False
            try:
                tmp = ij.IJ.openImage(image)
            except: 
                img_path = Path(image)
                tmp = ij.IJ.openImage(img_path)

        print(open_image)
        # Convert the image to Python format
        img = ij.py.from_java(tmp)
        print('image transferred to python')
        if open_image:
            tmp = ij.py.to_imageplus(img)
            tmp.changes = False
            tmp.close()
            del tmp
        elif not open_image:
            tmp.changes = False
            tmp.close()
            del tmp
        
        gc.collect()
        
        
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
    print('image concatenating')
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
            tmp_xarr = xarray.concat([tmp_xarr, tmp], dim="ch")

    return tmp_xarr