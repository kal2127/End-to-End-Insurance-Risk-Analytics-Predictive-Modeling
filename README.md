# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling

#  Task 1: Environment Setup & Exploratory Data Analysis (EDA) for AlphaCare Insurance Solutions (ACIS)

## 1. Business Context and Objective

[cite_start]We are a marketing analytics engineer team at **AlphaCare Insurance Solutions (ACIS)**[cite: 15, 16]. [cite_start]Our goal is to analyze historical car insurance claim data to **optimize marketing strategy** and **discover "low-risk" targets**[cite: 17]. [cite_start]Identifying these low-risk segments will allow ACIS to reduce premiums for them, creating an opportunity to attract new clients[cite: 17].

## 2. Technical Setup (Git and GitHub)

This section documents the foundational setup required for collaboration and version control:

* [cite_start]**Repository Initialization:** This repository was created to host all code, analysis, and reports for the challenge[cite: 162].
* [cite_start]**Initial Branch:** All work for this first stage is being developed on the **`task-1`** branch[cite: 163].
* [cite_start]**Version Control KPI:** We are adhering to the Key Performance Indicator (KPI) of committing our work at least three times a day with descriptive commit messages[cite: 166].
* [cite_start]**CI/CD Setup:** A foundational CI/CD pipeline using **GitHub Actions** has been configured to ensure code quality and integration standards (e.g., running linting/tests upon push/PR)[cite: 139].

## 3. Data Overview and Preparation

* [cite_start]**Data Source:** Historical insurance data from **February 2014 to August 2015**[cite: 48].
* **Key Variables for Analysis:** The analysis will focus on relationships between financial variables and risk drivers:
    * [cite_start]**Financial:** `TotalClaims`, `TotalPremium`[cite: 110, 111].
    * [cite_start]**Location:** `Province`, `PostalCode`[cite: 68, 69].
    * [cite_start]**Client:** `Gender`, `MaritalStatus`[cite: 64, 65].
    * [cite_start]**Vehicle:** `VehicleType`, `Make`, `Model`, `CustomValueEstimate`[cite: 75, 77, 78, 85].

## 4. Task 1.2: Project Planning - EDA & Statistical Focus

[cite_start]The objective of this analysis is to develop a foundational understanding of the data, assess its quality, and uncover initial patterns in risk and profitability[cite: 144, 145].

### Key Guiding Questions for EDA:

1.  [cite_start]**Loss Ratio:** What is the overall Loss Ratio (`TotalClaims` / `Total Premium`) for the portfolio, and how does it vary by `Province`, `VehicleType`, and `Gender`? [cite: 151, 152]
2.  [cite_start]**Outliers:** Are there outliers in `TotalClaims` or `CustomValueEstimate` that could skew our analysis? [cite: 153]
3.  [cite_start]**Temporal Trends:** Did the claim frequency or severity change over the 18-month period? [cite: 154]
4.  [cite_start]**Risk Segments:** Which vehicle makes/models are associated with the highest and lowest claim amounts? [cite: 155]

### Minimum Essential To Do:

* [cite_start]Perform **Descriptive Statistics** on numerical features[cite: 169].
* [cite_start]Review **Data Structure** and check for **missing values**[cite: 171, 173].
* [cite_start]Conduct **Univariate** (distributions) and **Bivariate/Multivariate** (correlations) analyses[cite: 174, 176].
* [cite_start]Produce **3 creative and beautiful plots** capturing key insights gained[cite: 183].

## 5. Next Steps

[cite_start]Upon completion of the EDA, the work will be merged into `main`, and we will proceed to **Task 2: Data Version Control (DVC)**[cite: 212].
