import pandas as pd
import numpy as np
from typing import Tuple

# Define the path to the raw data file
RAW_DATA_PATH = 'data/raw/data.csv'

def load_data(path: str) -> pd.DataFrame:
    """Loads the dataset from the specified path, using the correct delimiter."""
    try:
        # --- FIX: Added sep='|' to handle pipe-separated data ---
        df = pd.read_csv(path, sep='|')
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {path}. "
              "Please ensure your dataset is placed in this location.")
        return pd.DataFrame()


def perform_initial_assessment(df: pd.DataFrame) -> None:
    """
    Performs the initial assessment (df.info(), df.describe(), missing values)
    as required for Task 1.2.
    """
    if df.empty:
        return

    print("\n" + "="*50)
    print("TASK 1.2: INITIAL DATA ASSESSMENT")
    print("="*50)

    # 1. Data Structure Review (df.info())
    print("\n--- 1. Data Structure (Types and Non-Null Counts) ---")
    df.info()

    # 2. Descriptive Statistics (df.describe())
    print("\n--- 2. Descriptive Statistics for Numerical Features ---")
    # Transpose (.T) for easier reading
    print(df.describe(include=np.number).T)
    
    # Also include categorical column summary for completeness
    print("\n--- 3. Top Categories and Counts (Categorical Features) ---")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        # Print the value counts for the top 5 categories
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(5))

    # 4. Data Quality Assessment: Missing Values
    print("\n--- 4. Missing Value Analysis ---")
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    missing_percentage = (missing_counts / total_rows) * 100

    # Filter to show only columns with at least one missing value
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentage.round(2)
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )

    if missing_info.empty:
        print("✅ No missing values found in the dataset.")
    else:
        print(missing_info)
        print("\n*Action required: Address these missing values in the next cleaning step.*")


if __name__ == "__main__":
    # Ensure you have placed your CSV file at 'data/raw/insurance_data.csv'
    # before running this script.
    
    print("Starting data loading and initial assessment...")
    
    insurance_df = load_data(RAW_DATA_PATH)
    
    if not insurance_df.empty:
        perform_initial_assessment(insurance_df)
    
    print("\nInitial assessment complete.")