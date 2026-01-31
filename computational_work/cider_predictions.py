from localcider.sequenceParameters import SequenceParameters
import numpy as np
import pandas as pd

aromatics=['F','Y','W']
positives=['R','H','K']
negatives=['D','E']
valines=['V','V']

def countAA(sequence,group):
    count=0
    for aa in sequence:
        if aa in group:
            count +=1
    return count

def get_sequence_parameters(sequence: str) -> dict:
    FCR=SequenceParameters(sequence).get_FCR()
    NCPR=SequenceParameters(sequence).get_NCPR()
    mean_hydropathy=SequenceParameters(sequence).get_mean_hydropathy()
    kappa= SequenceParameters(sequence).get_kappa()
    SCD=SequenceParameters(sequence).get_SCD()
    fraction_disorder_promoting=SequenceParameters(sequence).get_fraction_disorder_promoting()
    countPos=SequenceParameters(sequence).get_countPos()
    countNeut=SequenceParameters(sequence).get_countNeut()
    countNeg=SequenceParameters(sequence).get_countNeg()
    countAro=countAA(sequence,aromatics)
    isoelectric_point=SequenceParameters(sequence).get_isoelectric_point()
    Omega=SequenceParameters(sequence).get_Omega()
    valine=countAA(sequence,valines)

    features = {'sequence':[sequence],
                'FCR':[FCR],
                'NCPR':[NCPR],
                'mean_hydropathy':[mean_hydropathy],
                'kappa': [np.nan] if kappa == -1 else [kappa] ,
                'SCD':SCD,
                'fraction_disorder_promoting':[fraction_disorder_promoting],     
                'countPos':[countPos],
                'countNeut':[countNeut],
                'countNeg':[countNeg],
                'countAro':[countAro],
                'isoelectric_point':[isoelectric_point],
                'Omega':[Omega],
                'valines':[valine]
               }
    return features

def evaluate_sequence_features(sequences: list[str])-> pd.DataFrame:
    df = pd.DataFrame()
    for i in sequences:
        new_df = pd.DataFrame(get_sequence_parameters(i))
        df = pd.concat([df, new_df], ignore_index=True)
    return df

def get_cider_predictions(df, sequence_column):
    sequences = df[sequence_column].unique().tolist()
    sequence_df = evaluate_sequence_features(sequences)
    df = pd.merge(df, sequence_df)
    return df