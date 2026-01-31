import glob 
import pandas as pd
import matplotlib.pyplot as plt

def import_sequence_files(csv_files, seq_df):
    tmp_df = pd.DataFrame()
    for file in csv_files:
        csv_df = pd.read_csv(file)
        csv_df['sequence'] = csv_df['sequence'].str.replace('\n', '', regex=False)
        current_sequence = csv_df['sequence'].iloc[0]
        seq_row = seq_df.loc[seq_df['sequence'] == current_sequence]
        
        merge_df = pd.merge(seq_row, csv_df, on='sequence', how='left')
        tmp_df = pd.concat([tmp_df, merge_df])

    return tmp_df

def get_files_list(path):
    return glob.glob(path+ r'\*.csv')

def get_sequence_df(path_csv, path_seq):
    seq_df = pd.read_csv(path_seq)
    csv_files = get_files_list(path_csv)
    return import_sequence_files(csv_files, seq_df)

if __name__ == '__main__':
    df = get_sequence_df(r'D:\OneDrive\Virus\E1A_albatros_predictions', r'D:\OneDrive\Virus\Sequences\E1A_tilepositions.csv')

