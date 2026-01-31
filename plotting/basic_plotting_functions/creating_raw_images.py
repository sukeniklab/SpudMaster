from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

def is_list_of_lists(item):
    return isinstance(item, list) and all(isinstance(element, list) for element in item)

def make_cmaps(color_name:list[str], color_gradients: list[list[tuple]], N_step=256):
    cmap_list = []
    if type(color_name) != list:
        color_name = [color_name]
    if not is_list_of_lists(color_gradients):
        color_gradients = [color_gradients]
    for name, color in zip(color_name, color_gradients):
        cmap = mcolors.LinearSegmentedColormap.from_list(name, color, N=N_step)
        cmap_list.append(cmap)
    return cmap_list  

def get_vscale(image):
    img_values = image.values
    
    vmax = np.max(img_values)
    vmin = np.min(img_values)

    return Normalize(vmin, vmax)

def calculate_pixel_fraction(scale_bar_size, x, pixel_scale)-> float:
    micron_x = x/pixel_scale
    scalebar_scale = scale_bar_size/micron_x
    return scalebar_scale

def set_scalebar(scale_size, image_x, pixel_scale):
    scale_fraction = calculate_pixel_fraction(scale_size, image_x, pixel_scale)
    scalebar = ScaleBar(1/pixel_scale, "um", 
                        frameon=False,
                        color='white', 
                        location= "lower right", 
                        fixed_value=100,
                        scale_loc="none",
                        fixed_units="um", pad=0.1, border_pad=0.05,
                        width_fraction=75/image_x)
    return scalebar