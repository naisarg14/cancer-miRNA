import pandas as pd
import sys, os
from tqdm import tqdm
import glob

def load_expression_data(file_path, sample_id=None):
    expression_df = pd.read_csv(file_path, sep="\t", skiprows=7)
    expression_df = expression_df.rename(columns={"635nm": "Expression"})
    expression_df = expression_df[["G_Name", "G_ID", "Expression"]]
    threshold, mean_blank = mean_threshold(expression_df)
    expression_df = expression_df[expression_df["G_Name"].str.startswith("hsa")]

    new_rows = pd.DataFrame([{"G_Name": "threshold", "G_ID": "threshold", "Expression": threshold},
                             {"G_Name": "mean_blank", "G_ID": "mean_blank", "Expression": mean_blank}])
    expression_df = pd.concat([expression_df, new_rows], ignore_index=True)
    if sample_id is not None:
        expression_df = expression_df.rename(columns={"Expression": sample_id})
    return expression_df

def mean_threshold(df):
    blank = df[df['G_Name'].str.contains("BLANK")]['Expression'].astype(float)
    sorted_blank = blank.sort_values()
    top_cutoff = max(int(0.05 * len(sorted_blank)), 1)
    filtered_blank = sorted_blank[top_cutoff:-top_cutoff]
    
    mean_blank = filtered_blank.mean()
    std_blank = filtered_blank.std(ddof=0)
    threshold = mean_blank + 2 * std_blank

    return threshold, mean_blank

def main():
    label_db = pd.read_csv("geo_gse_files/combined.csv")
    data_matrix = None
    folder = sys.argv[1]
    matrix_name = folder.removesuffix("/").replace("_RAW", "_matrix.csv").replace("downloads", "matrix")
    files = glob.glob(os.path.join(folder, "*.txt"))

    for file_name in tqdm(files):
        sample_id = os.path.basename(file_name).split("_")[0]
        if sample_id in label_db["GEO accession"].values:
            expression_df = load_expression_data(file_name, sample_id)
            repeated_g_ids = expression_df[expression_df.duplicated(subset=["G_ID"], keep=False)]
            for g_id in repeated_g_ids["G_ID"].unique():
                duplicates = expression_df[expression_df["G_ID"] == g_id]
                for i, idx in enumerate(duplicates.index):
                    expression_df.at[idx, "G_ID"] = f"{g_id} ({i+1})"
            
            label = label_db[label_db["GEO accession"] == sample_id]["Cancer Code"].values[0]
            new_row = pd.DataFrame([{"G_Name": "label", "G_ID": "label", sample_id: label}])
            expression_df = pd.concat([expression_df, new_row], ignore_index=True)

            if data_matrix is None:
                data_matrix = expression_df
            else:
                data_matrix = pd.merge(data_matrix, expression_df, on=["G_Name", "G_ID"])

    if data_matrix is not None:
        data_matrix.to_csv(matrix_name, index=False)


if __name__ == "__main__":
    main()