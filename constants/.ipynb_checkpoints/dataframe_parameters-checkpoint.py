# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:30:49 2025

@author: patri
"""

from constants import columns_names
import pandas as pd

def init_image_analysis_df():
    tmp_dic = {value: ['']
               for i, (name,value) in enumerate(vars(columns_names).items())
               if not name.startswith('__')}

    df = pd.DataFrame(tmp_dic)
    return df       
         