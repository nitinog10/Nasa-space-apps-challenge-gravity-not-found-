"""
Fix data issues in the features_engineered.csv file
Removes infinities and extremely large values
"""

import pandas as pd
import numpy as np
from pathlib import Path

def fix_data_issues():
    """Fix infinity and large value issues in the engineered features"""
    
    print("Loading engineered features...")
    data_path = Path('data/processed/features_engineered.csv')
    df = pd.read_csv(data_path)
    
    print(f"Original shape: {df.shape}")
    
    # Check for infinity values
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    inf_cols = inf_mask.any()
    inf_cols = inf_cols[inf_cols].index.tolist()
    
    if inf_cols:
        print(f"\nFound infinity values in {len(inf_cols)} columns:")
        for col in inf_cols[:5]:  # Show first 5
            print(f"  - {col}")
        if len(inf_cols) > 5:
            print(f"  ... and {len(inf_cols) - 5} more")
    
    # Replace infinity with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Check for extremely large values
    for col in numeric_cols:
        if df[col].dtype in [np.float64, np.float32]:
            # Replace extremely large values with NaN
            max_val = np.finfo(np.float32).max / 1000  # Conservative threshold
            df.loc[df[col].abs() > max_val, col] = np.nan
    
    # Fill NaN values with appropriate methods
    print("\nFilling missing values...")
    
    # For rolling features, use forward fill then backward fill
    rolling_cols = [col for col in df.columns if 'rolling' in col]
    if rolling_cols:
        df[rolling_cols] = df[rolling_cols].fillna(method='ffill').fillna(method='bfill')
    
    # For lag features, use forward fill
    lag_cols = [col for col in df.columns if 'lag' in col]
    if lag_cols:
        df[lag_cols] = df[lag_cols].fillna(method='ffill')
    
    # For remaining numeric columns, use median
    remaining_na = df[numeric_cols].isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    
    if len(remaining_na) > 0:
        print(f"\nFilling remaining NaN values in {len(remaining_na)} columns with median...")
        for col in remaining_na.index:
            median_val = df[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
    
    # Final check
    na_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"\nAfter cleaning:")
    print(f"  - NaN values: {na_count}")
    print(f"  - Infinity values: {inf_count}")
    print(f"  - Final shape: {df.shape}")
    
    # Save cleaned data
    output_path = Path('data/processed/features_engineered_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved cleaned data to: {output_path}")
    
    # Also overwrite the original file
    df.to_csv(data_path, index=False)
    print(f"✓ Updated original file: {data_path}")
    
    return df

if __name__ == "__main__":
    fix_data_issues()
    print("\n✓ Data cleaning complete! You can now run the training again.")
