# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:55:10 2025

@author: patri
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from pandas.api.types import CategoricalDtype
from scipy import stats
from scipy.stats import zscore
from calculations.significance import set_pvalue
from IPython.display import clear_output

def pixels_to_micron(df: pd.DataFrame, area_column: str='area', pixels_per_micron: float = 3.0769) -> pd.DataFrame:
    df= df.rename(columns={area_column: 'area_pixels'})
    df['area_micron'] = df['area_pixels']/pixels_per_micron
    return df


def area_cutoffs(df: pd.DataFrame, area = 'area_micron', lower_cutoff:float = 600.0, upper_cutoff: float=7000)-> pd.DataFrame:
    df = df.loc[(df[area] > lower_cutoff) & (df[area] < upper_cutoff)]
    df = df.reset_index(drop=True)
    return df


def remove_maximums(df: pd.DataFrame, channel: str, value_cutoff: float = 55000)-> pd.DataFrame:
    channel_cutoff = channel+'_intensity_max'
    df = df.loc[df[channel_cutoff] < value_cutoff]
    df = df.reset_index(drop=True)
    return df 


def calculate_time(df: pd.DataFrame, experiment: str): 
    experiment_timeframe ={
        'infection' : 4,
        'osmoticperturbation' : 0.1,
        'heatshock': 0.5,
        'blasticidin': 0.5,
    }
    
    timeframe = experiment_time[experiment]
    df['time'] = df['timepoint'].astype(int) * timeframe
    return df


def remove_minimums(df: pd.DataFrame, channel: str, value_cutoff = 400)-> pd.DataFrame:
    channel_cutoff = channel+'_intensity_min'
    df = df.loc[df[channel_cutoff] > value_cutoff]
    df = df.reset_index(drop=True)
    return df


def directacceptor_mean_cutoffs(df: pd.DataFrame, lower_cutoff: float = 1000, upper_cutoff=40000) -> pd.DataFrame:
    df = df.loc[(df['directAcceptor_intensity_mean']> lower_cutoff) & (df['directAcceptor_intensity_mean'] < upper_cutoff)]
    return df


def donor_mean_cutoffs(df: pd.DataFrame, lower_cutoff: float = 600, upper_cutoff=20000) -> pd.DataFrame:
    df = df.loc[(df['donor_intensity_mean']> lower_cutoff) & (df['donor_intensity_mean'] < upper_cutoff)]
    return df


def infection_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    # Define threshold conditions for each timepoint
    thresholds = {
        0: 400, 
        1: 2000, 
        2: 7000,
        3: 15000,  # Extend for other timepoints if needed
    }
    
    # Apply filtering conditionally for timepoints in the dictionary
    for time, threshold in thresholds.items():
        df = df.loc[~((df['timepoint'] == time) & (df['mCherry_intensity_mean'] > threshold))]

    df = df.reset_index(drop=True)
    return df

def infection_parameters(df: pd.DataFrame, group_values=['construct', 'timepoint', 'date', 'well'], lower_cutoff=0.15, upper_cutoff = 0.85) -> pd.DataFrame:
    df = df.copy()
    df['experimentparameter'] = 'mixed_population'
    
    tmp_df_holder = pd.DataFrame()

    for _, group in df.groupby(group_values):
        lower_threshold = group['mCherry_intensity_mean'].quantile(lower_cutoff)
        upper_threshold = group['mCherry_intensity_mean'].quantile(upper_cutoff)

        group = group.copy()
        group.loc[group['mCherry_intensity_mean'] <= lower_threshold, 'experimentparameter'] = 'noninfected'
        group.loc[group['mCherry_intensity_mean'] >= upper_threshold, 'experimentparameter'] = 'infected'

        tmp_df_holder = pd.concat([tmp_df_holder, group], ignore_index=True)

    return tmp_df_holder


def infection_cutoffs(df: pd.DataFrame, lower_cutoff=3000, upper_cutoff = 10000) -> pd.DataFrame:
    df = df.copy()

    noninfected =df.loc[df['mCherry_intensity_mean'] < lower_cutoff]
    mixed = df.loc[(df['mCherry_intensity_mean'] >= lower_cutoff) & (df['mCherry_intensity_mean'] <= upper_cutoff)]
    infected = df.loc[df['mCherry_intensity_mean'] > upper_cutoff]

    noninfected.loc[:, 'experimentparameter'] = 'noninfected'
    mixed.loc[:, 'experimentparameter'] = 'mixed_population'
    infected.loc[:, 'experimentparameter'] = 'infected'

    return pd.concat([noninfected, mixed, infected])

def set_categorical_order(df: pd.DataFrame, column_name: str, category_order: list[str]) -> pd.DataFrame:
    df = df.copy()  # Ensure we don't modify the original DataFrame in-place

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Ensure existing values in the column are not lost
    existing_values = df[column_name].dropna().unique()
    
    # Combine user-defined order with existing unique values while preserving order
    complete_order = list(dict.fromkeys(category_order + list(existing_values)))  

    # Apply the categorical dtype
    cat_type = CategoricalDtype(categories=complete_order, ordered=True)
    df[column_name] = df[column_name].astype(cat_type)

    return df  # Return modified DataFrame without affecting other columns

def sort_values(df: pd.DataFrame, sort_dict: dict[str, List[str]])-> pd.DataFrame:
    tmp_df = df.copy()
    for index, sorter_values in sort_dict.items():
        tmp_df = set_categorical_order(tmp_df, index, sorter_values)
    return tmp_df

def remove_outliers(df, group_values, check_column='Efret', sub_groupers= ['date', 'well'], observed=False, as_index=False, std_dev = 2):
    """
    Remove outliers from a DataFrame based on median values outside the interquartile range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to process
    group_values : list or str
        Column(s) to group by for outlier detection
    check_column : str
        Column to check for outliers
    observed : bool, optional (default=True)
        Whether to use only observed values in groupby
    as_index : bool, optional (default=False)
        Whether to return groupby result as index
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    filtered_dfs = []

    # Loop through each group defined by group_values
    for name, group in df.groupby(group_values, observed=True, as_index=False):
        # Compute median 'Efret' per 'date' and 'well'
        median_df = group.groupby(sub_groupers, as_index=False)[check_column].median().dropna()
        
        # Calculate the z-score of the median values
        median_df['zscore'] = zscore(median_df[check_column])
        # Compute absolute zscore and flag groups where abs(zscore) > 2
        median_df['abs_zscore'] = median_df['zscore'].abs()
        median_df['outlier'] = median_df['abs_zscore'] > std_dev
      
        new_groupers = sub_groupers + ['abs_zscore', 'outlier']
        # Merge the outlier flag back to the original group on 'date' and 'well'
        group = group.merge(median_df[new_groupers], on=['date', 'well'], how='left')
        
            
        # Optionally, if you want to drop the groups flagged as outliers
        group_filtered = group[group['outlier'] == False]
        
        
        filtered_dfs.append(group_filtered)
        
    #display(abs(zscore(median_efret)) > 2)
        
        
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df
    
def get_stats(df, group_values, stats= ['mean', 'median', 'min', 'max', 'std', 'count', 'var', 'list'] , stat_column='Efret', observed=True, as_index=False):
    """
    Compute specified statistics for a given column after grouping by specified columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_values (list): Columns to group by.
    - stats (list): List of statistics to compute. Default includes 'mean', 'median', 'min', 'max', 'std', 'count', and 'var'.
    - stat_column (str): The column on which to compute the statistics.
    - observed (bool): If True, only observed values for categorical groupers are shown. Default is True.
    - as_index (bool): If True, the group labels are used as the index. Default is False.
    - append_all_values (bool): If True, appends a column with all values of the stat_column. Default is True.

    Returns:
    - pd.DataFrame: DataFrame with the computed statistics.
    """
    if type(stats) != list:
        stats = [stats]
    grouped_data = df.groupby(group_values, observed=observed, as_index=as_index)[stat_column]
    
    
    stats_dic = {'mean': grouped_data.mean(),
                 'median': grouped_data.median(),
                 'min': grouped_data.min(),
                 'max': grouped_data.max(),
                 'std': grouped_data.std(),
                 'var': grouped_data.var(),
                 'count': grouped_data.count(),
                 'list': grouped_data.apply(list)
                }
                
    
    init_df = None
    
    for stat in stats:
        tmp_data = stats_dic[stat]
        if type(init_df) != pd.DataFrame:
            init_df = tmp_data
            init_df = init_df.rename(columns={stat_column: stat_column+'_'+stat})

        else:
            tmp_data = tmp_data.rename(columns={stat_column: stat_column+'_'+stat})
            tmp_data = tmp_data.drop(columns=group_values)
            init_df = pd.concat([init_df, tmp_data], axis=1)

    init_df = init_df.reset_index(drop=True)
    return init_df

def delta_time(df, delta_column, group_values=['construct', 'experiment'], comparison: int = 1):
    df_copy = df.copy()

    if df_copy[delta_column].dtype == list:
        
        df_copy[delta_column] = df_copy[delta_column].apply(np.array)
        df_copy['delta_'+delta_column] = df_copy[delta_column] - df_copy.groupby(group_values, observed=False)['Efret_median'].transform(lambda x: x.iloc[comparison])
    
    else:    
    
        df_copy['delta_'+delta_column] = df_copy[delta_column] - df_copy.groupby(group_values, observed=False)[delta_column].transform(lambda x: x.iloc[comparison])
        df_copy['delta_'+delta_column] = df_copy['delta_'+delta_column].fillna(0)

    return df_copy

def delta_list_median(df, delta_df, delta_column, group_values=['construct', 'experiment'], comparison: int =1):
    df_copy = df.copy()
    df_copy[delta_column] = df_copy[delta_column].apply(np.array)

    # Use nth(1) safely
    delta_medians = (
        delta_df.groupby(group_values, observed=False)['Efret_median']
        .apply(lambda x: x.iloc[comparison] if len(x) > 1 else np.nan)
        .reset_index()
    )
    delta_medians.rename(columns={'Efret_median': 'Efret_median_delta'}, inplace=True)


    # Merge the median values onto df_copy based on group_values
    df_copy = df_copy.merge(delta_medians, on=group_values, how='left')
          
    # Subtract median from each element of the list column
    df_copy['delta_' + delta_column] = df_copy.apply(
        lambda row: row[delta_column] - row['Efret_median_delta'] 
        if np.isfinite(row['Efret_median_delta']) else np.nan, 
        axis=1
    )

    # Drop the extra merged column if not needed
    df_copy.drop(columns=['Efret_median_delta'], inplace=True)
    
    return df_copy
"""
def delta_list_median(df, delta_df, delta_column, group_values= ['construct', 'experiment']):
    
    df_copy = df.copy()
    df_copy[delta_column] = df_copy[delta_column].apply(np.array)
    '''
    #delta_copy = delta_df.copy()
    
    #df_copy['delta_'+delta_column] = df_copy[delta_column] - delta_copy.groupby(group_values, observed=False)['Efret_median'].transform('second')
    '''
    delta_medians = delta_df.groupby(group_values, observed=False)['Efret_median'].nth(1).reset_index()
    print(delta_medians)
    # Merge the median values onto df_copy based on group_values
    df_copy = df_copy.merge(delta_medians, on=group_values, how='left', suffixes=('', '_delta'))
          
    # Subtract median from each element of the list column
    df_copy['delta_' + delta_column] = df_copy.apply(
        lambda row: row[delta_column] - row['Efret_median_delta'] 
        if np.isfinite(row['Efret_median_delta']) else np.nan, 
        axis=1
    )

    # Drop the extra merged column if not needed
    df_copy.drop(columns=['Efret_median_delta'], inplace=True)
    
    return df_copy
"""    

def remove_controls(df, control_1 = 'mTQ2', control_2='mNG'):
    return df.loc[(df['construct']!= control_1) & (df['construct']!=control_2)]

def drop_counts(df, count=20):
    df = df.loc[df['Efret_count'] > count]
    return df.reset_index(drop=True)


def get_population_significance(df, group_values, sign_column='Efret_median', observed=False,
                                 as_index=False, equal_var=True, nan_policy='propagate',
                                 comparison_value='Growth', comparison_column='experimentparameter'):
    
    groupby_df = df.groupby(group_values, observed=observed, as_index=as_index)
    sign_df = pd.DataFrame()
    comparison_population = None  # Store this globally so other groups can compare to it

    for group_key, group_data in groupby_df:
        # Convert group_key to dict using group_values
        group_dict = dict(zip(group_values, group_key if isinstance(group_key, tuple) else (group_key,)))

        # Extract comparison value
        current_comp_val = group_data[comparison_column].unique()[0]

        # If this group is the comparison group
        if current_comp_val == comparison_value:
            comparison_population = group_data[sign_column].values
            ttest, pvalue, star = 'NaN', 'NaN', 'ns'
        else:
            if comparison_population is None:
                raise ValueError(f"Comparison population for '{comparison_value}' not found before other group.")
            variable_population = group_data[sign_column].values
            ttest, pvalue = stats.ttest_ind(comparison_population, variable_population,
                                            equal_var=equal_var, nan_policy=nan_policy)
            star = set_pvalue(pvalue)

        # Build result dictionary
        result = {k: group_dict[k] for k in group_values}
        result.update({
            comparison_column: current_comp_val,
            'ttest': ttest,
            'pvalue': pvalue,
            'star_value': star
        })

        # Append to result dataframe
        sign_df = pd.concat([sign_df, pd.DataFrame([result])], ignore_index=True)

    return sign_df