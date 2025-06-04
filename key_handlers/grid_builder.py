# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:32:42 2025

@author: patri
"""

from key_handlers.wellbuilder import WellBuilder
import numpy as np



def build_rectangular_well(pattern='96-wells', custom_pattern = None, custom_column= None) -> list(): 
    valid_pattern = {'96-wells', '8-wells', 'custom'}
    if pattern is not valid_pattern:
        raise ValueError(f'pattern must be set to {valid_pattern}')
    if pattern =='custom' and custom_pattern == 'None':
        raise ValueError('Custom is selected, please add in a custom pattern: such as 1x1')
    
    builder = WellBuilder()
    wells =   builder.get_wells
    
    :
        
    return wells
    
    
    
builder = WellBuilder()
wells = builder.get_wells()
   
    
def build_polygon_well()