import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import plotter_functions as pf
from constants import plotting_paramters as pps
from plotting.basic_plotting_functions import utils

def _set_scatter_and_errorbars(ax, x, y, yerr, marker_size= pps.experiment_median, marker_color = pps.experiment_median_color, zorder = pps.zorder, elinewidth = pps.eline_width):
    '''
    ax[row].scatter(x= df['plot_position'], y= df['delta_Efret_median'], s=4, c='gray', zorder=3)
    ax[row].errorbar(x= df['plot_position'], y= df['delta_Efret_median'], yerr= df['std_median_per_timepoint'], linestyle='none', color='k',elinewidth=4)
    '''
    ax.scatter(x= x, y= y, s=marker_size, c=experiment_median_color, zorder=zorder)
    ax.errorbar(x= x, y= y, yerr= yerr, linestyle='none', color='k', elinewidth= elinewidth)
    
    return ax

def create_comparison_violin(ax, df):


    

def overlay_violin_plot(df:pd.DataFrame(), variables: list, figsize=None, rows=None, cols = None, dims='long'):
    
    unique_parameters, nunique_parameters = pf.get_unique_parameters(df , groupby_values)
    nunique_spot_1 = nunique_parameters[variables[0]]
    if len(variables) > 1:
        nunique_spot_2 = nunique_parameters[variables[1]]
        row_len = int(np.floor(nunique_spot_1/nunique_spot_2))
    else:
        nunique_spot_2 = 4
        row_len = nunique_spot_1
    #set up figure size
    if figsize == None: 
        fig_multipliers = utils.dim_multiplier(dims)
        width, length = nunique_spot_2*fig_multipliers[0], nunique_spot_1*fig_multipliers[1]
        figsize = (width, length) 

    if rows != None:
        row_len = rows
    if cols != None:
        col_len = cols
    
    fig, ax = plt.subplots(row_len, col_len, figsize=figsize)
    ax = ax.flatten()
        
    group_generator = new_tmp.groupby(grouped_values, observed=True)

    for index, container in enumerate(group_generator):
        current_index = index/len(grouped_values)
        tmp_df = container[1]

        

    plt.tight_layout