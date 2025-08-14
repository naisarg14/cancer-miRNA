import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm

def blank_correction(matrix_df):
    thresholds = matrix_df[matrix_df["G_Name"] == "threshold"].iloc[0, 1:].astype(float)
    mean_blanks = matrix_df[matrix_df["G_Name"] == "mean_blank"].iloc[0, 1:].astype(float)
    label = matrix_df[matrix_df["G_Name"] == "label"]


    rna_df = matrix_df[~matrix_df["G_Name"].isin(["threshold", "mean_blank", "label"])].copy()
    rna_df.iloc[:, 1:] = rna_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    for col in tqdm(rna_df.columns[1:], desc="Performing blank correction"):
        col_threshold = thresholds[col]
        col_mean_blank = mean_blanks[col]
        rna_df[col] = rna_df[col].apply(lambda x: None if float(x) < col_threshold else x - col_mean_blank)

    return rna_df, label

def filter_na_rows(df, threshold=0.95):
    max_na_count = int(threshold * df.shape[1])
    rows_to_remove = df.apply(lambda row: row.isnull().sum() > max_na_count, axis=1)
    removed_row_names = df.loc[rows_to_remove, "G_Name"].tolist()
    filtered_df = df.loc[~rows_to_remove]
    
    return filtered_df, removed_row_names

    
def normalize_internal_controls(df):
    target_genes = ['hsa-miR-149-3p', 'hsa-miR-2861', 'hsa-miR-4463']

    control_rows = df[df["G_Name"].isin(target_genes)]
    overall_mean = np.nanmean(control_rows.iloc[:, 1:].values)
    column_mean = control_rows.iloc[:, 1:].mean(axis=0, skipna=True)

    normalized_df = df.copy()
    for col in tqdm(df.columns[1:], desc="Normalizing data based on internal controls"):
        normalized_df[col] = df[col].where(
            pd.isna(df[col]),
            df[col] + overall_mean - column_mean[col]
        )

    return normalized_df


def main():
    matrix_file = sys.argv[1]
    geo_id = matrix_file.split("/")[-1].replace("_matrix.csv", "").replace(".csv", "")
    os.makedirs(geo_id, exist_ok=True)
    matrix_df = pd.read_csv(matrix_file, low_memory=False)
    matrix_df.drop(columns=["G_ID"], inplace=True)
    
    #Step-0: Blank correction and Filter NA rows
    blank_corrected_df, label = blank_correction(matrix_df)
    blank_csv = f"{geo_id}/{geo_id}_blank_corrected.csv"
    blank_corrected_df.to_csv(blank_csv, index=False)

    filtered_df, removed_row_names = filter_na_rows(blank_corrected_df, threshold=0.95)
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.to_csv(f"{geo_id}/{geo_id}_filtered.csv", index=False)

    with open(f"{geo_id}/{geo_id}_removed_rna.txt", "w") as f:
        for name in removed_row_names:
            f.write(f"{name}\n")

    #Step-1: Log transformation
    log_df = filtered_df.copy()
    for col in tqdm(log_df.columns[1:], desc="Processing columns"):
        log_df[col] = np.log2(log_df[col].fillna(0.1).astype(float))
    log_df.to_csv(f"{geo_id}/{geo_id}_log_transformed.csv", index=False)

    #Step-2: Internal Contorl Normalization
    normalized_df = normalize_internal_controls(log_df)
    normalized_df.to_csv(f"{geo_id}/{geo_id}_internal_control_normalized.csv", index=False)

    #Step-3: Label DF
    #result_df = pd.concat([normalized_df, label], axis=0)  
    normalized_df.loc[len(normalized_df)] = label.iloc[0]
    normalized_df.reset_index(drop=True, inplace=True)
    normalized_df.to_csv(f"{geo_id}/{geo_id}_processed.csv", index=False)

    #print(result_df)

if __name__ == "__main__":
    main()