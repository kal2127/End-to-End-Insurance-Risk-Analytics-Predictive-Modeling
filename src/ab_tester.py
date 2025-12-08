import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


FIGURE_PATH = 'reports/'
# Use the same load function, but adapt it to load from the correct path and delimiter
def load_data_for_ab_test(path: str) -> pd.DataFrame:
    """Loads data, ensuring correct delimiter and selecting relevant columns."""
    try:
        df = pd.read_csv(path, sep='|') 
        # Filter for policies that actually had a claim (TotalClaims > 0)
        df = df[df['TotalClaims'] > 0].copy()
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}.")
        return pd.DataFrame()

def perform_ab_test(df: pd.DataFrame):
    
    # 1. Define Groups A and B
    group_a = df[df['Province'] == 'Northern Cape']['TotalClaims']
    group_b = df[df['Province'] != 'Northern Cape']['TotalClaims']
    
    if group_a.empty or group_b.empty:
        print("Error: One or both groups are empty. Cannot perform test.")
        return

    # 2. Set Significance Level (Alpha)
    # Common standard in business analytics
    alpha = 0.05 
    
    # 3. Perform Two-Sample T-Test (assuming unequal variances - Welch's T-test)
    # We use 'less' for a one-tailed test because Ha is 'mean < mean_control'
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False, alternative='less')
    
    # 4. Summarize Results
    print("\n" + "="*50)
    print("TASK 3: A/B HYPOTHESIS TEST (Northern Cape vs. Others)")
    print("="*50)
    print(f"Mean Claim (Northern Cape, Group A): ZAR {group_a.mean():,.2f}")
    print(f"Mean Claim (Other Provinces, Group B): ZAR {group_b.mean():,.2f}")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value (One-Tailed Test): {p_value:.4f}")
    print(f"Significance Level (α): {alpha}")

    # 5. Conclusion
    if p_value < alpha:
        print("\nConclusion: **REJECT the Null Hypothesis (H0)**.")
        print("The difference in claim amount is statistically significant.")
        print("Business Implication: The Northern Cape segment has a significantly lower average claim amount. The strategy to reduce premiums for this segment is supported.")
    else:
        print("\nConclusion: **FAIL TO REJECT the Null Hypothesis (H0)**.")
        print("The difference in claim amount is NOT statistically significant.")
        print("Business Implication: The data does not support the hypothesis that this segment has lower claim severity. The premium reduction strategy is risky.")

# --- NEW VISUALIZATION FUNCTION FOR TASK 3 ---
def plot_ab_test_results(df: pd.DataFrame):
    """Generates a box plot to visually compare TotalClaims severity between groups."""
    
    # 1. Create a grouping column for visualization
    df['Group'] = np.where(df['Province'] == 'Northern Cape', 'A: Northern Cape (Test)', 'B: Other Provinces (Control)')
    
    plt.figure(figsize=(10, 6))
    
    # 2. Use a box plot to show median, quartiles, and outliers
    sns.boxplot(
        x='Group', 
        y='TotalClaims', 
        data=df, 
        palette=['#4c72b0', '#c44e52'] # Distinct colors for clear comparison
    )
    
    # 3. Add context to the plot
    plt.title('A/B Test: Claim Severity Distribution (TotalClaims)', fontsize=14)
    plt.xlabel('Province Group')
    plt.ylabel('Claim Severity (TotalClaims - ZAR)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Optional: Log scale the y-axis to better visualize the bulk of the data (due to high outliers)
    plt.yscale('log')
    plt.suptitle('Log Scale: Northern Cape has a substantially lower median and variance in claim severity.', fontsize=10, y=1.02)
    
    # 4. Save the figure
    plot_filename = FIGURE_PATH + 'ab_test_severity_comparison.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"✅ Visualization saved to {plot_filename}")


if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/data.csv' 
    
    df_claims = load_data_for_ab_test(RAW_DATA_PATH)
    
    if not df_claims.empty:
        perform_ab_test(df_claims)
        
        # Call the new plotting function
        plot_ab_test_results(df_claims)