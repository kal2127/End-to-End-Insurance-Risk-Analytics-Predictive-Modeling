import pandas as pd
import numpy as np
import io
import joblib # New import for saving models
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.tools import add_constant

# Constants
RAW_DATA_PATH = 'data/raw/data.csv'
REPORT_PATH = 'reports/model_summary.txt'
MODEL_PATH = 'models/'
FIGURE_PATH = 'reports/' # Ensure this directory exists
SAMPLE_SIZE = 200000

# --- Helper Functions (Same as before) ---

def downcast_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces memory usage by downcasting integer and float columns."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def load_and_preprocess_data(path: str) -> pd.DataFrame:
    """Loads, cleans, and engineers features for modeling."""
    df = pd.read_csv(path, sep='|', low_memory=False) 
    df = downcast_data_types(df) 
    
    # Random Sampling to prevent MemoryError
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} rows for modeling due to memory constraints.")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    # 1. Feature Engineering (Target Variables)
    df['Claim_Indicator'] = np.where(df['TotalClaims'] > 0, 1, 0)
    df['Claim_Count'] = df['Claim_Indicator'] 
    df['Claim_Severity'] = np.where(df['Claim_Indicator'] == 1, df['TotalClaims'], np.nan)

    # 2. Select Relevant Features
    features = [
        'Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'RegistrationYear'
    ]
    df_model = df[features + ['Claim_Count', 'Claim_Severity', 'TotalPremium']].copy()
    
    # 3. Clean Missing Values (Fixed FutureWarning)
    df_model['CustomValueEstimate'] = df_model['CustomValueEstimate'].fillna(df_model['CustomValueEstimate'].median())
    
    return df_model

def build_preprocessor():
    """Defines the feature engineering pipeline (ColumnTransformer)."""
    numerical_features = ['CustomValueEstimate', 'RegistrationYear']
    categorical_features = ['Province', 'Gender', 'VehicleType']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# --- Model Building Functions (Updated to return preprocessor) ---

def build_frequency_model(df_model: pd.DataFrame, output_stream):
    # ... (function body remains the same, except for return statement)
    output_stream.write("\n" + "="*80 + "\n")
    output_stream.write("--- Training Frequency Model (Poisson GLM) ---\n")
    output_stream.write("="*80 + "\n")
    
    df_freq = df_model.dropna(subset=['CustomValueEstimate', 'RegistrationYear']).copy()
    X = df_freq[['Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'RegistrationYear']]
    y = df_freq['Claim_Count']

    preprocessor = build_preprocessor()
    
    X_processed = preprocessor.fit_transform(X) 
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    X_processed_df = add_constant(X_processed_df, prepend=True, has_constant='add')
    
    poisson_model = GLM(y, X_processed_df, family=families.Poisson()).fit()
    output_stream.write(poisson_model.summary().as_text())
    
    return poisson_model, preprocessor

def build_severity_model(df_model: pd.DataFrame, freq_preprocessor, output_stream):
    # ... (function body remains the same)
    output_stream.write("\n" + "="*80 + "\n")
    output_stream.write("--- Training Severity Model (Gamma GLM) ---\n")
    output_stream.write("="*80 + "\n")
    
    df_sev = df_model.dropna(subset=['Claim_Severity']).copy()
    X = df_sev[['Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'RegistrationYear']]
    y = df_sev['Claim_Severity']

    X_processed = freq_preprocessor.transform(X)
    feature_names = freq_preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    X_processed_df = add_constant(X_processed_df, prepend=True, has_constant='add')
    
    gamma_model = GLM(y, X_processed_df, family=families.Gamma(link=families.links.Log())).fit()
    output_stream.write(gamma_model.summary().as_text())
    
    return gamma_model

def calculate_pure_premium(df_model: pd.DataFrame, freq_model, sev_model, freq_preprocessor, output_stream):
    # ... (function body remains the same for calculation)
    output_stream.write("\n" + "="*80 + "\n")
    output_stream.write("--- Final Pure Premium Calculation and Comparison ---\n")
    output_stream.write("="*80 + "\n")
    
    X = df_model[['Province', 'Gender', 'VehicleType', 'CustomValueEstimate', 'RegistrationYear']]
    
    X_processed = freq_preprocessor.transform(X)
    feature_names = freq_preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    X_processed_df = add_constant(X_processed_df, prepend=True, has_constant='add')

    df_model['Predicted_Frequency'] = freq_model.predict(X_processed_df)
    df_model['Predicted_Severity'] = sev_model.predict(X_processed_df)
    df_model['Pure_Premium'] = df_model['Predicted_Frequency'] * df_model['Predicted_Severity']
    
    # Build the report string
    report = f"Portfolio Total Actual Premium: ZAR {df_model['TotalPremium'].sum():,.2f}\n"
    report += f"Portfolio Total Modeled Pure Premium: ZAR {df_model['Pure_Premium'].sum():,.2f}\n"
    
    northern_cape_metrics = df_model[df_model['Province'] == 'Northern Cape']
    report += "\nNorthern Cape Comparison:\n"
    report += f"  Actual Premium Collected: ZAR {northern_cape_metrics['TotalPremium'].sum():,.2f}\n"
    report += f"  Modeled Pure Premium: ZAR {northern_cape_metrics['Pure_Premium'].sum():,.2f}\n"
    
    output_stream.write(report)
    
    # Return the dataframe with predictions for plotting
    return df_model 


# --- NEW FUNCTION FOR VISUALIZATIONS AND MODEL SAVING ---

def save_artifacts_and_plot_diagnostics(df_model, freq_model, sev_model, preprocessor):
    """Saves models and creates required diagnostic plots."""
    
    # 1. Save Model Artifacts (.pkl)
    joblib.dump(freq_model, MODEL_PATH + 'poisson_frequency_model.pkl')
    joblib.dump(sev_model, MODEL_PATH + 'gamma_severity_model.pkl')
    joblib.dump(preprocessor, MODEL_PATH + 'feature_preprocessor.pkl')
    print(f"✅ Models saved to {MODEL_PATH}")
    
    # 2. Actual vs. Predicted Plot (Overall Pure Premium)
    plt.figure(figsize=(12, 6))
    
    # Use log scale due to heavy right skew in claims and premium
    sns.scatterplot(x=np.log1p(df_model['TotalPremium']), 
                    y=np.log1p(df_model['Pure_Premium']), 
                    alpha=0.1, 
                    color='blue')
    
    # Plot y=x line (perfect prediction)
    max_val = max(np.log1p(df_model['TotalPremium'].max()), np.log1p(df_model['Pure_Premium'].max()))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title('Actual Premium vs. Predicted Pure Premium (Log Scale)')
    plt.xlabel('Log(Actual TotalPremium)')
    plt.ylabel('Log(Predicted PurePremium)')
    plt.legend()
    plt.savefig(FIGURE_PATH + 'actual_vs_predicted.png')
    plt.close()
    print(f"✅ Actual vs Predicted plot saved.")

    # 3. Residual Distribution Plot
    # Calculate Residuals: Total Claims - Pure Premium
    df_model['Residuals'] = df_model['Claim_Count'] * df_model['Claim_Severity'].fillna(0) - df_model['Pure_Premium']
    
    plt.figure(figsize=(8, 6))
    sns.histplot(df_model['Residuals'], bins=50, kde=True, color='purple', log_scale=(False, True))
    plt.title('Distribution of Model Residuals')
    plt.xlabel('Residual (Actual Claims - Predicted Pure Premium)')
    plt.ylabel('Count (Log Scale)')
    plt.savefig(FIGURE_PATH + 'residual_distribution.png')
    plt.close()
    print(f"✅ Residual Distribution plot saved.")
    
    # 4. Feature Importance Chart (Gamma Severity Model Coefficients)
    # The Gamma model was stable, so we use its coefficients.
    coefficients = pd.DataFrame({
        'Coefficient': sev_model.params.index,
        'Value': sev_model.params.values
    })
    # Filter out the constant and small, uninteresting coefficients
    coefficients = coefficients[coefficients['Coefficient'] != 'const']
    coefficients['Abs_Value'] = coefficients['Value'].abs()
    coefficients = coefficients.sort_values('Abs_Value', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Coefficient', data=coefficients, palette='viridis')
    plt.title('Top 10 Feature Importance (Gamma Severity Model Coefficients)')
    plt.xlabel('Coefficient Value (Higher means greater Severity)')
    plt.ylabel('Feature')
    plt.savefig(FIGURE_PATH + 'feature_importance_severity.png')
    plt.close()
    print(f"✅ Feature Importance plot saved.")

# --- Main Execution Block (Updated to include new functions) ---

if __name__ == '__main__':
    
    output_stream = io.StringIO()
    output_stream.write("Statistical Modeling Report (Task 4)\n")
    output_stream.write("Generated using GLM (Poisson for Frequency, Gamma for Severity)\n\n")

    df_clean = load_and_preprocess_data(RAW_DATA_PATH)
    
    if not df_clean.empty:
        # Step 1: Train Frequency Model and get the fitted preprocessor
        poisson_model, freq_preprocessor = build_frequency_model(df_clean, output_stream)
        
        # Step 2: Train Severity Model (using the fitted preprocessor)
        gamma_model = build_severity_model(df_clean, freq_preprocessor, output_stream)
        
        # Step 3: Calculate Final Pure Premium and get the DF
        df_results = calculate_pure_premium(df_clean, poisson_model, gamma_model, freq_preprocessor, output_stream)

        # Step 4: Save Artifacts and Plot Diagnostics (NEW STEP)
        save_artifacts_and_plot_diagnostics(df_results, poisson_model, gamma_model, freq_preprocessor)
        
        # Step 5: Save the captured output to the report file
        with open(REPORT_PATH, 'w') as f:
            f.write(output_stream.getvalue())

        print(f"\n✅ All Task 4 Deliverables Complete.")
        print(f"Check 'models/' for .pkl files and 'reports/figures/' for .png files.")