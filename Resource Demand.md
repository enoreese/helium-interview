# Resource Demand Documentation
## 1. Overview
This document outlines the steps taken to build predictive models for forecasting healthcare resource demand. The models are designed to predict future visit counts for each institution based on historical patterns and relevant features.

## 2. Feature Engineering
### 2.1 Aggregate Features
The first step involves aggregating features from the provided datasets to create meaningful representations of healthcare institution activity. The features extracted are used as inputs for subsequent predictive modeling.
- Group data by institution and date
- Aggregate features such as visit count, unique patients count, in-patients count, out-patients counts, etc.
- Create a new dataframe 'visit_data' with aggregated features

### 2.2 Temporal and Lag Features
Temporal features and lag variables are created to capture time dependencies and trends in the data. Resampling the data at a daily frequency and filling missing values ensures a consistent time series.
- Create temporal features such as year, month, day, and day of the week
- Lag features are introduced by shifting visit counts to create lag-1, lag-2, and lag-3
- Target variables t1, t2, and t3 are created by shifting visit counts backward
- Ensure appropriate handling of missing values and data alignment

## 3. Data Splitting
- Split the time series data into training and testing sets
- Maintain stratification based on the 'institution' column
- Use 70% of the data for training

## 4. Model Training and Evaluation
### 4.1 Regression
- Define a list of regression models including Linear Regression, XGBoost, Lasso, Ridge, and RandomForest
- For each model, preprocess the data using ColumnTransformer
  - One-hot encode categorical columns
  - Standard scale numerical columns
- Train the model on the training set and evaluate on the testing set

### 4.2 Evaluation
- Metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) are reported for both training and testing sets

## 5. Hyperparameter Tuning
- Define hyperparameter space for XGBoost using hyperopt library
- Implement objective function for hyperparameter tuning
- Use hyperopt's Tree Parzen Estimator (TPE) algorithm for optimization

## 6. Model Explainability and Interpretation
### 6.1 SHAP Values Overview
SHAP (SHapley Additive exPlanations) values provide a way to explain the output of machine learning models. 
They allocate contributions of each feature to the prediction for each instance, providing insights into how each feature influences the model's output. 
Here, we use SHAP values to interpret the resource demand prediction model.

### 6.2 SHAP Summary Plot
The SHAP summary plot serves as a replacement for the traditional bar chart of feature importance. 
It offers a comprehensive view of each feature's importance and the range of effects over the dataset. 
The plot indicates which features are most influential and how changes in their values impact the predicted demand.

In our model:

- The primary factor influencing demand forecasts is the 'number of unique patients.'
- The next most powerful indicator is the 'last 3 days' demand.

### 6.3 SHAP Dependence Plot
SHAP dependence plots illustrate how the model output varies with changes in a specific feature's value. 
Each dot in the plot represents an individual instance in the dataset, allowing for the observation of trends and interactions.

### 6.4 Interpretation and Insights
- **Primary Influencers**: Understanding that the 'visit count' is the most significant factor allows healthcare providers to focus on strategies to attract and retain patients.
- **Temporal Patterns**: Observing the impact of 'last 3 days' demand provides insights into the short-term nature of resource demand. Providers can use this information for more agile and responsive resource allocation.

## 7. Conclusion
This documentation provides a comprehensive overview of the process involved in developing predictive resource demand models. It covers data aggregation, feature engineering, model selection, and hyperparameter tuning. The resulting models are ready for deployment and can be integrated into the healthcare system for real-time resource demand predictions.

