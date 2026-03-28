# Exploratory Data Analysis & Machine Learning Report

## 1. Overview
The dataset contains **5000** rows and **18** columns. Our goal was to predict the binary target `high_value_purchase`. During EDA, it was observed that some missing values existed in features like `avg_review_rating` representing incomplete customer profiles. 

## 2. Data Cleaning & Feature Engineering
- **Missing Values**: Handled using Median Imputation for numerical features and 'Most Frequent' (Mode) for categorical features.
- **Categorical Encoding**: Handled via One-Hot Encoding.
- **Numerical Scaling**: Normalized numeric distributions via `StandardScaler` to handle magnitude differences.
- **Engineered Features**:
  1. `total_spend`: Captured overall monetary value (`total_purchases` * `avg_order_value`).
  2. `engagement_score`: Captured user interaction intensity based on reviews and email open rates.

## 3. Model Training & Comparison
We evaluated the following traditional ML models using an 80/20 train-test split:

| Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |      0.754 |    0.75813  |    0.746 |   0.752016 |  0.817884 |
| Random Forest       |      0.795 |    0.810526 |    0.77  |   0.789744 |  0.890072 |
| Gradient Boosting   |      0.815 |    0.835821 |    0.784 |   0.809082 |  0.897924 |

**Selected Model**: `Gradient Boosting`  
It achieved the best performance metric (F1-Score: **0.8091**), maintaining a good balance between Precision and Recall.

## 4. Feature Insights
### Top 10 Important Features

| Feature | Importance |
|---|---|
| avg_order_value | 0.3903 |
| account_age_months | 0.1040 |
| days_since_last_purchase | 0.0892 |
| total_purchases | 0.0745 |
| customer_segment_Gold | 0.0713 |
| total_spend | 0.0629 |
| customer_segment_Platinum | 0.0610 |
| customer_segment_Bronze | 0.0561 |
| customer_segment_Silver | 0.0397 |
| bounce_rate | 0.0144 |


## 5. Conclusion
- The predictions for the hold-out test set are saved in `predictions.csv`.
- The visualizations generated from the EDA and Model Evaluation are safely stored within the `plots/` directory.
- `Gradient Boosting` and `Random Forest` consistently identify core monetary features like `total_spend` and `avg_order_value` as massive predictors for predicting high-value purchasers!
