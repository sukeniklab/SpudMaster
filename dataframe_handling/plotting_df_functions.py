import os
import pandas as pd
import numpy as np
from calculations import fret_calculations, processing_cleanup
import scipy.stats as stats
from calculations.significance import set_pvalue
import matplotlib.patches as mpatches
from dataframe_handling import dataframe_setup as dfs
from typing import Tuple, List, Union
#figure out how to import all the dataframes in a directory

from constants import dataframe_parameters

def compute_osmotic_delta_efret(
    df: pd.DataFrame,
    group_values: List[str] = ['construct', 'date', 'experimentcondition', 'replicate'],
    timepoints: Union[List[int], Tuple[int, int]] = (0, 1),
    efret_column: str = 'Efret',
    label_column: str = 'label'
) -> pd.DataFrame:
    """
    Compute the change in Efret between two timepoints within grouped subsets of the DataFrame.

    Parameters:
    - df: Input DataFrame containing Efret values and metadata
    - group_values: List of columns to group by before timepoint comparison
    - timepoints: A tuple or list of exactly two timepoints (before, after)
    - efret_column: Name of the Efret measurement column
    - label_column: Column used to align cells between timepoints

    Returns:
    - A new DataFrame with 'delta_Efret' column added to both timepoint slices
    """
    if len(timepoints) != 2:
        raise ValueError("Exactly two timepoints must be specified.")
    
    time_before, time_after = timepoints
    new_df = pd.DataFrame()
    df = df.copy()

    grouped = df.groupby(group_values, as_index=False)

    for key, group in grouped:
        df_before = group[group['timepoint'] == time_before].sort_values(by=label_column).reset_index(drop=True)
        df_after = group[group['timepoint'] == time_after].sort_values(by=label_column).reset_index(drop=True)

        if df_before.empty or df_after.empty:
            print(f"Skipping group {key}: missing data at timepoints {time_before} or {time_after}")
            continue

        # Safety check: ensure labels align
        if not df_before[label_column].equals(df_after[label_column]):
            raise ValueError(f"Label mismatch between timepoints for group {key}")

        # Compute delta Efret
        delta = df_after[efret_column].values - df_before[efret_column].values
        df_before['delta_Efret'] = 0.0  # optional: baseline
        df_after = df_after.copy()
        df_after['delta_Efret'] = delta

        new_df = pd.concat([new_df, df_before, df_after], ignore_index=True)

    return new_df.reset_index(drop=True)
        


def add_efret_clean_up_df(df, experiment, control_values= ['GS32', 'SED1', 'mTQ2', 'mNG'], pixel_conversion=3.0769, keep_middle_pop=False, *args, **kwargs):
    """
    converts given dtaframes into two dataframes based upon split values.

    Idea for creating control and experimental values dataframes.
    """
    #convert area of pixels into microns
    
    
    df = processing_cleanup.pixels_to_micron(df, area_column= kwarg.get('area_column', 'area'), pixels_to_microns=pixel_conversion)
    #calculate the fret efficiency based upon FRET column names
    df = fret_calculations.acceptor_correction(df, fret_stat = kwarg.get('fret_stat', 'mean'), crossex_correction= kwarg.get('crossex_correction', 0.068), bleedthrough_correction= kwarg.get('bleedthrough_correction', 0.47))
    df = fret_calculations.calculate_efret(df, fret_stat = kwarg.get('fret_stat', 'mean'))

    if experiment == 'osmotic':
        df = compute_osmotic_delta_efret(df, kwarg.get('osmotic_group_comparison_values', ['construct', 'date', 'experimentcondition', 'replicate']))
    
    #Removal of data with specific cutoffs
    df = processing_cleanup.area_cutoffs(df, area=kwargs.get('area_column', 'area_micron'),
        lower_cutoff=kwargs.get('area_lower_cutoff', 600.0),
        upper_cutoff=kwargs.get('area_upper_cutoff', 7000.0))
    df = processing_cleanup.remove_maximums(df, 'directAcceptor', value_cutoff=kwargs.get('max_cutoff', 55000))
    df = processing_cleanup.remove_minimums(df, 'directAcceptor', value_cutoff=kwargs.get('min_cutoff', 400))
    df = processing_cleanup.directacceptor_mean_cutoffs(df, lower_cutoff=kwargs.get('acceptor_mean_lower', 1000),
        upper_cutoff=kwargs.get('acceptor_mean_upper', 40000))
    df = processing_cleanup.donor_mean_cutoffs(df, lower_cutoff=kwargs.get('donor_mean_lower', 600),
        upper_cutoff=kwargs.get('donor_mean_upper', 20000))
    if not keep_middle_pop: 
        df = df.loc[['experimentparameter'] != 'mixed_population']
    
    return split_df_values(df, column = kwarg.get('split_column', 'construct'), values_to_left = control_values)


def filter_and_get_stats(df, full_group_list: list, column_drop_experients :list):
    """
    Filters acts to separate on experiment wide values. i.e,  getting experiment based means and medians vs. replicate based values to create replicate medians.
    """
    #list of column to remove for the list
    experiment_list = list(filter(lambda x: x not in column_drop_experients, full_group_list))
    #
    filtered_df = processing_cleanup.remove_outliers(df, experiment_list, 'Efret')
    #
    total_stats_df = processing_cleanup.get_stats(filtered_df, full_group_repeats)
    experiment_stats = processing_cleanup.get_stats(filtered_df, experiment_list)
    #
    return total_stats_df, experimental_stats

def check_column(df, column):
    """
    Makes sure the column exist within the dataframe
    """
    
    columns = df.columns
    if column in columns:
        return True
    else:
        raise ValueError(f"{column} is not found in df columns")

def create_list_columns(conden_df, merge_df, 
                        group_values: list = ['construct', 'experiment', 'experimentparameter', 'timepoint'], 
                        com_type: str='median'):
    """
    Creates unique columns based upon parameters given. First column is delta medians of the well for plotting individual scatter points
    The other is creating the full population list of list or population based list to plot the actual violin overlays.
    """
    col_dict = {'median': 'delta_median_wells',
           'list': 'delta_Efret_population'}
    try:
        new_column = col_dict[com_type]
    except: 
        raise ValueError(f"{com_type} is not a valid input. Update dictionary or try 'median' or 'list'" )
    column = 'delta_Efret_' + com_type

    if check_column(conden_df, column):
        consolidated_values = conden_df.groupby(group_values, observed=False, as_index=False)[column].apply(np.array)
        consolidated_values = consolidated_values.loc[(consolidated_values['construct']!= 'mTQ2') & (consolidated_values['construct']!= 'mNG')]
        consolidated_values = consolidated_values.dropna().reset_index(drop=True).rename(columns={column: new_column})
        return merge_df.merge(consolidated_values, on=group_values, how='left')