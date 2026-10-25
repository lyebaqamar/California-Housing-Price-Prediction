# California Housing Price Prediction
A machine learning project for predicting median house values in California districts using the 1990 California Census dataset. The project implements a complete workflow starting with custom NumPy-based math baselines (Normal Equation and Batch Gradient Descent) up to advanced spatial feature engineering and highly optimized Extreme Gradient Boosting (XGBoost).

This project was completed as **Task 3 of Month 1** during my Machine Learning journey, showcasing rigorous dataset exploration, clean architectural pipelines, and professional model evaluation.

📌 **Project Overview**
Predicting real estate prices is a classic regression problem. The objective of this project is to build a highly accurate, robust, and generalizable model that can predict the `median_house_value` of various California districts based on demographic, economic, and geographical coordinates.

Two primary execution phases were developed:
1. **NumPy baseline formulations** — Mathematical models built from scratch using the closed-form Normal Equation and Batch Gradient Descent (reaching ~64% R²).
2. **Ensemble & Advanced Gradient Boosting** — Machine learning pipelines utilizing geographical clustering (K-Means), distance mapping, and XGBoost (reaching a final **85.81% testing R²** and **84.39% prediction precision**).

🎯 **Objectives**
- Load and inspect the 1990 California Housing census dataset.
- Perform comprehensive Exploratory Data Analysis (EDA) and visualize target value distributions.
- Preprocess data cleanly (handle missing attributes using median imputation and categorical elements using drop-first dummy structures).
- Identify and treat continuous feature outliers using the Interquartile Range (IQR) Winsorization technique.
- Implement Standard Scaling and Min-Max Normalization algorithms from scratch using NumPy.
- Build a custom NumPy Linear Regression model using the analytical Normal Equation.
- Implement iterative parameter optimization using Batch Gradient Descent from scratch.
- Verify the custom implementations using Scikit-Learn's baseline models.
- Perform 10-Fold Cross-Validation to evaluate alternative algorithms (Ridge, Lasso, Elastic Net, Bayesian Ridge, Extra Trees).
- Engineer non-linear spatial cluster properties (K-Means on coordinates) and relative city distance indicators (LA & SF centers).
- Train and optimize an Extreme Gradient Boosting (XGBoost) regressor.
- Create interactive geographical visualizations using Plotly Express for live-app demonstrations.
- Serialize the champion pipeline into a production-ready model artifact.

📊 **Dataset**
The project utilizes the classic **California Housing dataset** derived from the 1990 U.S. Census.

### Dataset Characteristics
- **Total Samples**: 20,640 districts
- **Features**: 9 independent attributes (8 numerical, 1 categorical)
- **Target Variable**: `median_house_value` (continuous numeric value representing district home prices)
- **Dataset Constraints**: House values capped at a ceiling limit of $500,001.

### Attributes List
- `longitude` / `latitude`: Geographical spatial coordinates.
- `housing_median_age`: Median age of houses inside a district.
- `total_rooms` / `total_bedrooms`: Total counts of rooms and bedrooms within a block group.
- `population`: Total number of residents inside the block group.
- `households`: Total number of households in the district.
- `median_income`: Median household income (measured in tens of thousands of USD).
- `ocean_proximity`: Qualitative proximity category (categorical label).

🛠️ **Technologies & Libraries**
- Python
- NumPy (for array processing and from-scratch math)
- Pandas (for data manipulation and structure)
- Matplotlib & Seaborn (for static charts and performance plots)
- Plotly Express (for interactive virtual graphics)
- Scikit-learn (for pipeline pipelines, Cross-Validation, and baseline models)
- XGBoost (for advanced boosting modeling)
- Joblib (for model serialization and file saving)
- Google Colab (development workspace)

🔄 **Project Workflow**
```text
Dataset Loading
       ↓
Exploratory Data Analysis (EDA)
       ↓
NumPy Feature Preprocessing (Median Imputation, IQR Capping, One-Hot Encoding)
       ↓
Standard Scaling (Custom NumPy Formulas)
       ↓
Train / Test Matrix Splitting (80/20 Train-Test Setup)
       ↓
Normal Equation Solver & Batch Gradient Descent (NumPy Math from Scratch)
       ↓
Custom Models Evaluation & Scikit-Learn Verification
       ↓
10-Fold Cross-Validation Benchmark (Evaluated 5 Regression Algorithms)
       ↓
Advanced Spatial & Demographic Feature Engineering (K-Means, City Distances, Ratios)
       ↓
Champion XGBoost Regressor Implementation & Regularization Tuning
       ↓
Plotly Interactive Graphical Demonstrations (Spatial coordinates, Error Density plots)
       ↓
Production Pipeline Serialization & Export (housing_model.joblib)
###🔍 Exploratory Data Analysis The project conducts detailed statistical checks:

Checking dataset dimensions and column data types.
Running missing value analysis (uncovering 207 null cells inside total_bedrooms).
Finding variables correlating strongly with target home value (showing median_income leads at ~0.69 correlation).
Plotting distribution plots for the target variable to isolate boundary capping.
###⚙️ Data Preprocessing & Engineering

Data Leakage Mitigation: The train/test splitting was executed prior to numerical transformations.
Missing Values: Implemented median imputation on continuous variables to handle skewed features.
Outlier Treatment: Calculated IQR bounds on the training split and capped extreme numeric outliers to prevent model skewing.
Categorical Encoding: Designed custom matrix broadcast formulas to one-hot encode ocean_proximity categories cleanly.
Custom Scaling: Built custom standardization and normalization helper functions inside NumPy.
Advanced Feature Engineering:
Calculated direct Euclidean distances from coordinates to major centers: Los Angeles and San Francisco.
Created structural indicators: rooms_per_household, bedrooms_per_room, population_per_household, and income_per_capita.
Executed K-Means clustering (25 clusters) on latitude and longitude coordinates to capture micro-neighborhood parameters.
###🌲 Baseline Model — NumPy Regression from Scratch Two mathematical techniques were built to compute parameters without using high-level libraries:

Analytical Normal Equation: Calculates weights directly through closed-form matrix multiplication.
Batch Gradient Descent: Optimizes parameters iteratively over 10,000 iterations at a learning rate of 0.1.
Both baseline methods converged to identical weights, validating the precision of our raw NumPy code.

###🧠 Main Model — Regularized Spatial XGBoost To handle non-linear spatial boundaries, a tuned Extreme Gradient Boosting (XGBoost) regressor was implemented with heavy L1/L2 regularization (reg_alpha=10.0, reg_lambda=15.0) to secure stable generalizations.

###📊 Model Evaluation & Comparison

Model Stage	Evaluation Type	Metric	Performance Accuracy (% / Score)
1. NumPy Linear Regression	Test Subset	R² Accuracy	64.20%
2. Scikit-Learn LinearRegression	Test Subset	Intercept Alignment	Perfect Match (No Difference)
3. Ridge Regression	10-Fold CV	Mean RMSE	$68,904.26
4. Extra Trees Regressor	10-Fold CV	Mean RMSE	$51,300.17
5. Advanced XGBoost + Spatial Engineering	Test Subset	Testing R² Accuracy	85.81%
6. Advanced XGBoost + Spatial Engineering	Test Subset	Prediction Accuracy (100 - MAPE)	84.39%
###📉 Core Graphics & Interactive Plots

Interactive Spatial Plot: Displays a dynamic coordinate layout of California. Home pricing values are color-graded, while district population densities determine point sizes.
Feature Importance Matrix: Charts the contribution profiles of each feature, displaying the predictive dominance of your custom engineered spatial features.
Model Error Histogram: Displays model error distributions with integrated marginal boxes to ensure model residuals are normally distributed.
Progress Comparison Bar Chart: Summarizes performance progression, from your standard custom NumPy baseline to optimized ensemble models.
###💾 Model & Results Export The optimized champion pipeline was successfully serialized and compressed:

Serialized Model File: housing_model.joblib
This saved file packages the entire data preprocessing transformer column stack alongside the fitted model for seamless, single-command prediction setups.

👩‍💻 Author Laiba Qamar
BSCS Student |
Machine Learning Intern — Arch Technology

📚 Acknowledgements
Kaggle — California Housing Prices Dataset
Google Colab Workspace Environments
