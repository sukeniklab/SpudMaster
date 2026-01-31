import pandas as pd
from typing import List, Dict

def acceptor_correction(df: pd.DataFrame, fret_stat:str = 'mean', crossex_correction: float=0.15, bleedthrough_correction:float=0.48, mCh_subtract=0.173)-> pd.DataFrame:
    '''
    
    '''
    df['crossexcitation_correction'] = crossex_correction
    df['bleedthrough_correction'] = bleedthrough_correction
    if 'mCherry_intensity_' + fret_stat in df.columns:
        df['directAcceptor_intensity_'+fret_stat] = df['directAcceptor_intensity_'+fret_stat] - df['mCherry_intensity_'+fret_stat] * mCh_subtract
    
    df['acceptor_corrected_'+fret_stat] = df['acceptor_intensity_'+fret_stat] - df['donor_intensity_'+fret_stat]*bleedthrough_correction-df['directAcceptor_intensity_'+fret_stat] * crossex_correction
    return df

def calculate_efret(df: pd.DataFrame, stat: str='mean')->pd.DataFrame:
    df['Efret'] = df['acceptor_corrected_'+stat]/(df['donor_intensity_'+stat]+df['acceptor_corrected_'+stat])
    return df