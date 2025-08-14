import pandas as pd
import os

def remove_overlaps_from_gse113486():
    """
    Remove rows from GSE113486_data.csv if their GEO accession numbers 
    are present in GSE113740_data.csv
    """
    
    # File paths
    gse113486_path = "geo_gse_files/GSE113486_data.csv"
    gse113740_path = "geo_gse_files/GSE113740_data.csv"
    
    # Check if files exist
    if not os.path.exists(gse113486_path):
        print(f"Error: {gse113486_path} not found")
        return
    
    if not os.path.exists(gse113740_path):
        print(f"Error: {gse113740_path} not found")
        return
    
    try:
        # Read both CSV files
        print("Reading GSE113486_data.csv...")
        df_113486 = pd.read_csv(gse113486_path)
        print(f"  Rows in GSE113486: {len(df_113486)}")
        
        print("Reading GSE113740_data.csv...")
        df_113740 = pd.read_csv(gse113740_path)
        print(f"  Rows in GSE113740: {len(df_113740)}")
        
        # Check if GEO accession column exists in both files
        if 'GEO accession' not in df_113486.columns:
            print("Error: 'GEO accession' column not found in GSE113486_data.csv")
            return
        
        if 'GEO accession' not in df_113740.columns:
            print("Error: 'GEO accession' column not found in GSE113740_data.csv")
            return
        
        # Get GEO accessions from GSE113740
        gse113740_accessions = set(df_113740['GEO accession'].unique())
        print(f"  Unique GEO accessions in GSE113740: {len(gse113740_accessions)}")
        
        # Find overlapping accessions in GSE113486
        overlapping_accessions = df_113486[df_113486['GEO accession'].isin(gse113740_accessions)]['GEO accession'].unique()
        print(f"  Overlapping accessions found: {len(overlapping_accessions)}")
        
        if len(overlapping_accessions) > 0:
            print(f"  Overlapping accessions: {sorted(overlapping_accessions)}")
            
            # Remove rows with overlapping GEO accessions from GSE113486
            df_113486_filtered = df_113486[~df_113486['GEO accession'].isin(gse113740_accessions)]
            
            print(f"\nRemoving {len(df_113486) - len(df_113486_filtered)} rows from GSE113486")
            print(f"Rows remaining in GSE113486: {len(df_113486_filtered)}")
            
            # Create backup of original file
            backup_path = gse113486_path.replace('.csv', '_backup.csv')
            print(f"\nCreating backup: {backup_path}")
            df_113486.to_csv(backup_path, index=False)
            
            # Save the filtered data
            print(f"Saving filtered data to: {gse113486_path}")
            df_113486_filtered.to_csv(gse113486_path, index=False)
            
            print("\nOperation completed successfully!")
            print(f"Original file backed up as: {backup_path}")
            
        else:
            print("\nNo overlapping GEO accessions found. No changes made.")
        
        # Display summary
        print(f"\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Original GSE113486 rows: {len(df_113486)}")
        print(f"Overlapping rows removed: {len(overlapping_accessions)}")
        print(f"Final GSE113486 rows: {len(df_113486) - len(overlapping_accessions)}")
        
    except Exception as e:
        print(f"Error processing files: {e}")

def verify_removal():
    """
    Verify that the overlaps have been successfully removed
    """
    print(f"\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    
    try:
        # Read the files again to verify
        df_113486 = pd.read_csv("geo_gse_files/GSE113486_data.csv")
        df_113740 = pd.read_csv("geo_gse_files/GSE113740_data.csv")
        
        gse113740_accessions = set(df_113740['GEO accession'].unique())
        remaining_overlaps = df_113486[df_113486['GEO accession'].isin(gse113740_accessions)]
        
        if len(remaining_overlaps) == 0:
            print("✓ Verification successful: No overlapping GEO accessions remain")
        else:
            print(f"✗ Warning: {len(remaining_overlaps)} overlapping accessions still found:")
            print(remaining_overlaps['GEO accession'].unique())
            
    except Exception as e:
        print(f"Error during verification: {e}")

def main():
    print("Removing overlapping rows from GSE113486_data.csv...")
    print("="*50)
    
    remove_overlaps_from_gse113486()
    verify_removal()

if __name__ == "__main__":
    main()