import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RAW_DATA_PATH = 'data/raw/data.csv'
FIGURE_PATH = 'reports/figures/'

# --- Helper Function for Data Loading (Cleaned) ---
def load_and_clean_data(path: str) -> pd.DataFrame:
    """Loads data, using the correct delimiter, and converts TransactionMonth to datetime."""
    try:
        # Use the correct delimiter
        df = pd.read_csv(path, sep='|') 
        
        # Convert the TransactionMonth column to datetime objects
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}.")
        return pd.DataFrame()

# --- 3.1 Key Metric Calculation ---
def calculate_loss_ratio(df: pd.DataFrame) -> None:
    """Calculates and prints overall and grouped Loss Ratios."""
    
    # Calculate Overall Loss Ratio
    overall_loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    print(f"\n--- Overall Portfolio Loss Ratio: {overall_loss_ratio:.4f} ---")
    
    risk_factors = ['Province', 'VehicleType', 'Gender']
    
    for factor in risk_factors:
        grouped_data = df.groupby(factor).agg(
            TotalClaims=('TotalClaims', 'sum'),
            TotalPremium=('TotalPremium', 'sum')
        ).reset_index()
        
        grouped_data['LossRatio'] = grouped_data.apply(
            lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else 0,
            axis=1
        )
        print(f"\nLoss Ratio by {factor}:")
        print(grouped_data[[factor, 'LossRatio']].sort_values(by='LossRatio', ascending=False))

# --- 3.2 Univariate Analysis (Distributions) ---
def plot_univariate_distributions(df: pd.DataFrame) -> None:
    """Plots histograms for numerical and bar charts for categorical columns."""
    
    # Numerical Distributions (Histograms)
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Univariate Analysis: Numerical Distributions', fontsize=16)

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(FIGURE_PATH + 'univariate_distributions.png')
    plt.close() # Use plt.show() if running interactively

    # Categorical Distributions (Bar Charts)
    categorical_cols = ['Province', 'VehicleType', 'Gender']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Univariate Analysis: Categorical Counts', fontsize=16)

    for i, col in enumerate(categorical_cols):
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=axes[i])
        axes[i].set_title(f'Counts by {col}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(FIGURE_PATH + 'categorical_counts.png')
    plt.close()

# --- 3.3 Bivariate Analysis and Outlier Detection ---

def plot_outliers(df: pd.DataFrame) -> None:
    """Uses box plots to visualize outliers in key financial features."""
    
    financial_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[financial_cols])
    plt.title('Outlier Detection using Box Plots (Task 1 KPI)')
    plt.savefig(FIGURE_PATH + 'financial_outliers.png')
    plt.close()
    
def plot_temporal_trends(df: pd.DataFrame) -> None:
    """Plots temporal trends in claims, premium, and loss ratio."""
    
    # Group by month and calculate sums
    monthly_summary = df.groupby('TransactionMonth')[['TotalPremium', 'TotalClaims']].sum().reset_index()
    monthly_summary['LossRatio'] = monthly_summary['TotalClaims'] / monthly_summary['TotalPremium']
    
    # Plot trends over time
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Claims and Premium
    color = 'tab:blue'
    ax1.set_xlabel('Transaction Month')
    ax1.set_ylabel('Total Claims/Premium (ZAR)', color=color)
    ax1.plot(monthly_summary['TransactionMonth'], monthly_summary['TotalPremium'], label='Premium', color=color)
    ax1.plot(monthly_summary['TransactionMonth'], monthly_summary['TotalClaims'], label='Claims', linestyle='--', color='red')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Plot Loss Ratio on a secondary axis (Creative Plot 1)
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Loss Ratio', color=color) 
    ax2.plot(monthly_summary['TransactionMonth'], monthly_summary['LossRatio'], label='Loss Ratio', color=color, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Temporal Trend: Claims, Premium, and Loss Ratio')
    plt.savefig(FIGURE_PATH + 'temporal_trends.png')
    plt.close()


if __name__ == "__main__":
    
    print("Starting Task 1.2: Exploratory Data Analysis (EDA)...")
    
    # 1. Load and clean data (ensure date conversion)
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    
    if not df_clean.empty:
        # 2. Calculate Key Metrics
        calculate_loss_ratio(df_clean)
        
        # 3. Generate Plots (saved to reports/figures/)
        plot_univariate_distributions(df_clean)
        plot_outliers(df_clean)
        plot_temporal_trends(df_clean)
        
        print("\n✅ EDA analysis complete. Figures saved to 'reports/figures/'.")
        print("Review the Loss Ratio output to identify high/low risk segments.")
    else:
        print("EDA failed due to data loading error.")