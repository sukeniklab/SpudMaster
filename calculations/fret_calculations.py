import pandas as pd
from typing import List, Dict

def acceptor_correction(df: pd.DataFrame, stat:str = 'mean', crossex_correction: float=0.068, bleedthrough_correction:float=0.47)-> pd.DataFrame:
    '''
    
    '''
    df['crossexcitation_correction'] = crossex_correction
    df['bleedthrough_correction'] = bleedthrough_correction
    df['acceptor_corrected_'+stat] = df['acceptor_intensity_'+stat] - df['donor_intensity_'+stat]*bleedthrough_correction-df['directAcceptor_intensity_'+stat] * crossex_correction
    return df

def calculate_efret(df: pd.DataFrame, stat: str='mean')->pd.DataFrame:
    df['Efret'] = df['acceptor_corrected_'+stat]/(df['donor_intensity_'+stat]+df['acceptor_corrected_'+stat])
    return df