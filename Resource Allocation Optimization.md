# Resource Allocation Optimization

## Introduction
This document provides an overview and guidelines for the implementation of the resource allocation optimization algorithm (`resource_alloc.py`). The algorithm is designed to optimize the allocation of healthcare resources based on predicted demand for each institution. The implementation uses the PuLP library for linear programming in Python.

## Overview
### XGBoost Demand Prediction Model:
- The XGBoost model (`XGB_MODEL`) is loaded from a saved file (`'models/xgboost_t1_pipeline.sav'`).
- The model is then used to predict the demand for each institution using the function `get_forecast`.


### Resource Allocation Optimization Function:
The resource allocation optimization algorithm is implemented in the `optimize_resource_allocation` function. The function takes three main parameters:

- `demand_predictions`: A dictionary containing predicted demand for each institution. (From `get_forecast`)
- `available_resources`: A dictionary containing available resources for each institution.
- `constraints`: A dictionary containing additional constraints and objectives. containing additional constraints and objectives.

## Key Components
- **Decision Variables**: The decision variables represent the allocation of resources to each institution. These are defined using LpVariable from the pulp library.

- **Objective Function**: The objective is to maximize the total demand fulfillment, which is the sum of predicted demand multiplied by the allocation for each institution. The algorithm aims to allocate resources in a way that meets the predicted demand for each institution.

- **Constraints**:
  - ***Resource Constraints***: Ensure that the allocated resources do not exceed the available resources for each institution.
  - ***Total Allocation Limit***: Total allocation across all institutions should not exceed a specified value (e.g., 180). 
  - ***Max Allocation per Institution***: The allocation for each institution should not exceed a specified value (e.g., 3). 
  - ***Min Allocation per Institution***: The allocation for each institution should be at least a specified value (e.g., 1).
  - ***Demand Fulfillment Ratio***: The total demand fulfillment should be at least a specified percentage (e.g., 80%) of the total predicted demand.

- **Solving the Optimization Problem**: The linear programming problem is solved using the solve method from the pulp library.

- **Retrieving Results**: The final allocation results are obtained from the solved linear programming problem and returned as a dictionary.

## Guidelines for Implementation
1. **Input Data**:
   - Ensure that the input data for the XGBoost model and resource allocation optimization function is correctly loaded and formatted.

2. **Model and Column Definitions**:
   - Update model and column definitions based on the actual model and features used.

3. **Constraint Definitions**:
   - Define constraints based on the specific requirements and constraints of the healthcare resource allocation problem.

4. **Testing and Validation**:
   - Test the code with sample data to ensure it runs correctly.
   - Validate the results against expected outcomes.

By following these guidelines, the resource allocation optimization algorithm can be effectively implemented, ensuring accurate demand predictions and efficient resource allocation in healthcare institutions.