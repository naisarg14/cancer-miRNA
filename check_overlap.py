import pandas as pd
import os
from collections import defaultdict

# List of dataset files
files = ["GSE113740", "GSE164174", "GSE211692", "GSE212211", "GSE124158", "GSE112264", "GSE235028", "GSE106817"]

def extract_geo_accessions():
    """Extract GEO accession numbers from all CSV files and check for overlaps"""
    
    # Dictionary to store GEO accessions for each dataset
    dataset_accessions = {}
    all_accessions = []
    
    # Process each file
    for file in files:
        csv_path = f"geo_gse_files/{file}_data.csv"
        
        if os.path.exists(csv_path):
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                
                # Extract GEO accession column
                if 'GEO accession' in df.columns:
                    accessions = set(df['GEO accession'].unique())
                    dataset_accessions[file] = accessions
                    all_accessions.extend(accessions)
                    
                    print(f"\n{file}:")
                    print(f"  Total GEO accessions: {len(accessions)}")
                    print(f"  Sample accessions: {list(accessions)[:5]}...")
                    
                else:
                    print(f"Warning: 'GEO accession' column not found in {csv_path}")
                    
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"File not found: {csv_path}")
    
    return dataset_accessions, all_accessions

def check_overlaps(dataset_accessions):
    """Check for overlaps between datasets"""
    
    print("\n" + "="*50)
    print("OVERLAP ANALYSIS")
    print("="*50)
    
    datasets = list(dataset_accessions.keys())
    overlaps_found = False
    
    # Check pairwise overlaps
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            dataset1 = datasets[i]
            dataset2 = datasets[j]
            
            overlap = dataset_accessions[dataset1].intersection(dataset_accessions[dataset2])
            
            if overlap:
                overlaps_found = True
                print(f"\nOverlap between {dataset1} and {dataset2}:")
                print(f"  Number of overlapping accessions: {len(overlap)}")
                print(f"  Overlapping accessions: {sorted(list(overlap))}")
    
    if not overlaps_found:
        print("\nNo overlaps found between any datasets.")
    
    # Summary statistics
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_unique = len(set().union(*dataset_accessions.values()))
    total_all = sum(len(accessions) for accessions in dataset_accessions.values())
    
    print(f"Total GEO accessions across all datasets: {total_all}")
    print(f"Unique GEO accessions: {total_unique}")
    print(f"Duplicate accessions: {total_all - total_unique}")
    
    return overlaps_found

def main():
    print("Extracting GEO accession data from CSV files...")
    print("="*50)
    
    dataset_accessions, all_accessions = extract_geo_accessions()
    
    if dataset_accessions:
        overlaps_found = check_overlaps(dataset_accessions)
        
        # Create a detailed overlap report
        if overlaps_found:
            print(f"\n" + "="*50)
            print("DETAILED OVERLAP REPORT")
            print("="*50)
            
            # Count occurrences of each accession
            accession_count = defaultdict(list)
            for dataset, accessions in dataset_accessions.items():
                for acc in accessions:
                    accession_count[acc].append(dataset)
            
            # Find duplicates
            duplicates = {acc: datasets for acc, datasets in accession_count.items() if len(datasets) > 1}
            
            if duplicates:
                print(f"\nDuplicate GEO accessions found in multiple datasets:")
                for acc, datasets in sorted(duplicates.items()):
                    print(f"  {acc}: appears in {datasets}")
    
    else:
        print("No valid data found in any CSV files.")

if __name__ == "__main__":
    main()