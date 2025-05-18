import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from inmoose.pycombat import pycombat_norm
from tqdm import tqdm

def main():
    if len(sys.argv) < 3:
        raise ValueError("Please provide the file paths for the data files")
    
    dfs = []
    batch_labels = []
    file_list = sys.argv[1:]
    print(file_list)
    for i, file in tqdm(enumerate(file_list), desc="Reading data files", total=len(file_list)):
        df = pd.read_csv(file)
        batch_label = f"Batch-{i+1}"
        dfs.append(df)
        batch_labels.extend([batch_label] * (df.shape[1] - 1))

    combined_df = dfs[0]
    for df in tqdm(dfs[1:], desc="Combining dataframes"):
        combined_df = combined_df.merge(df, on='G_Name', how='outer')

    #drop rows and columns with all NaN values
    label_row = combined_df[combined_df['G_Name'] == 'label']
    combined_df = combined_df[combined_df['G_Name'] != 'label']
    combined_df.dropna(how='any', inplace=True)
    combined_df.dropna(axis=1, how='all', inplace=True)

    #convert batch label to numbers
    combined_df.set_index('G_Name', inplace=True)
    encoder = LabelEncoder()
    batch_labels_encoded = encoder.fit_transform(batch_labels)

    #PyComBat
    numeric_data = combined_df.apply(pd.to_numeric, errors='coerce')
    microarray_corrected = pycombat_norm(numeric_data, batch_labels_encoded)
    corrected_df = pd.DataFrame(microarray_corrected, index=combined_df.index, columns=combined_df.columns)

    #add labels back
    corrected_df.reset_index(inplace=True)
    label_row.reset_index(drop=True, inplace=True)
    corrected_with_labels = pd.concat([corrected_df, label_row], ignore_index=True)
    corrected_with_labels.to_csv('matrix/batch_corrected.csv', index=False)

    #for training
    transpose_df = pd.read_csv('matrix/batch_corrected.csv', low_memory=False)
    transpose_df = transpose_df.T
    transpose_df.columns = transpose_df.iloc[0]
    transpose_df = transpose_df[1:]
    transpose_df.index.name = 'Samples'
    transpose_df.to_csv('matrix/training.csv')
    

if __name__ == "__main__":
    main()
