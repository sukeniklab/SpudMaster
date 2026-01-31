import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_unique_parameters(df, parameters):
    parameter_dict = {}
    len_par_dict = {}
    for parameter in parameters:
        uni_parms = df[parameter].unique()
        uni_parms_len = len(uni_parms)
        parameter_dict[parameter] = uniq_parms
        len_par_dict[parameter] = uni_parms_len
    return parameter_dict, len_par_dict

def get_ylims(df=df, column='delta_Efret' ylims: tuple(int, int) = None) -> tuple(int, int):
    if ylims != None:
        ymax = df[column].max()

def set_vline(ax, x):
    return ax.axvline(x=x, c='k', linestyle='--')

def set_hline(ax, y):
    return ax.axhline(y=y, c='k', linestyle='--')

def set_star_ypositions(nunique_parms: int, ymax: int, value_adjustment) -> list[int]:
    star_y = []
    for index, n in enumerate(nunique_parms):
        y = ymax* value_ajustment - index * 0.007
        star_y.append(y)

    return star_y

def return_list_positions(possible_positions, current_poisitons):
    new_list = []
    for pos in current_position:
        index_value = possible_positions.index(pos)
        new_list.append(possible_list[index_value])

    return new_list

def print_star_value_text(x, y, df):
    star_val = df.loc[df['plot_position']==x, 'star_value']
    ax.text(x, y_pos, s=s.iloc[0], fontsize=8, fontweight='bold', ha='center')
    
    return 