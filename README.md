# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling

Project Summary

This project focuses on analyzing mobility insurance data to understand customer risk, identify profitable segments, and build a predictive model for estimating Total Claims.
It follows an MLOps-style pipeline using Git for code versioning and DVC for data versioning.

📂 Project Structure
├── data/
│   ├── raw/                # Raw input data (tracked with DVC)
│   ├── processed/          # Cleaned & transformed data
├── src/
│   ├── data_processor.py   # Data loading, cleaning, EDA
│   ├── hypothesis_tests.py # A/B testing scripts (Task 3)
│   ├── model.py            # Statistical/ML models (Task 4)
├── reports/
│   ├── figures/            # Plots generated during EDA
│  
├── .dvc/                   # DVC metadata
├── .gitignore
├── requirements.txt
└── README.md

🎯 Business Objective

The goal is to help the insurer:

Reduce portfolio Loss Ratio

Identify low-risk customer segments

Improve pricing strategy through predictive modeling

Target profitable customers while controlling high-risk exposure

🎯 Project Goals

Perform EDA on 1M+ rows of mobility insurance data

Calculate Loss Ratio by region, vehicle type, and customer segment

Conduct A/B Hypothesis Tests for low-risk vs. portfolio performance

Build a Generalized Linear Model (GLM) to predict Total Claims

Implement reproducible MLOps workflow using Git + DVC

🛠️ Tools & Technologies
Tool	Purpose
Python	Programming language
Pandas, NumPy	Data manipulation
Matplotlib, Seaborn	Visualization
Git, GitHub	Code versioning
DVC	Data versioning
venv	Virtual environment
🚀 Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/insurance-risk-analytics.git
cd insurance-risk-analytics

2. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

3. Install dependencies
pip install -r requirements.txt

4. Pull the dataset using DVC
dvc pull

📊 Exploratory Data Analysis (Task 1)

The dataset contains 1,000,098 rows and 53 columns, including:

Customer demographics

Vehicle details

Premium and claims

Risk classifications

Financial metrics

Key Insights

Portfolio Loss Ratio = 1.0623 → Portfolio is unprofitable

Highest risk segments:

Gauteng Province (LR = 1.2247)

Heavy Commercial Vehicles (LR = 1.6284)

Lowest risk segments:

Northern Cape (LR = 0.2827)

Bus category (LR = 0.1373)

All visualizations are saved in reports/figures/.

📦 Data Version Control (Task 2)
DVC Initialization
dvc init

Track raw data
dvc add data/raw/data.csv

Add remote storage
dvc remote add -d localstorage .dvc_store

Commit to Git
git add .
git commit -m "Added DVC and tracked raw dataset"

📈 Task 3: Statistical Testing ( upcoming )

Objective:
Test whether claim severity in Northern Cape is significantly lower than the national average using:

t-tests

Effect size measurement

Confidence intervals

🤖 Task 4: Predictive Modeling ( upcoming )

Planned models:

Generalized Linear Model (GLM)

Possibly Random Forest or Gradient Boosting

Performance measured using:

RMSE

MAE

R²

Goal: Predict TotalClaims and optimize pricing recommendations.

📝 Reporting

Deliverables to be created:

Interim Report

Final Report

Medium Article summarizing:

EDA insights

Statistical test results

Predictive model performance

Business implications

📧 Contact

Author: Kalkidan Tesfaye
Email: 27kalkidan21@gmail.com
GitHub: https://github.com/kal2127

