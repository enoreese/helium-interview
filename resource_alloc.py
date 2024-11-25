import pandas as pd
import numpy as np
import joblib
import random
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

XGB_MODEL = joblib.load('models/xgboost_t1_pipeline.sav')
XGB_COLUMNS = ['institution', 'inst_type', 'month', 'dayofweek',
               'visit_count', 'no_unique_patients', 'no_out_patients',
               'no_in_patients', 'in_out_ratio', 'avg_age', 'avg_male_age',
               'avg_female_age', 'max_age', 'min_age', 'no_male',
               'no_female', 'no_unique_states', 'day', 'lag_1', 'lag_2', 'lag_3']


def optimize_resource_allocation(demand_predictions, available_resources, constraints):
    """
    Optimize the allocation of healthcare resources based on demand predictions.

    Parameters:
    - demand_predictions: Dictionary containing predicted demand for each institution.
    - available_resources: Dictionary containing available resources for each institution.
    - constraints: Dictionary containing constraints and objectives.

    Returns:
    - allocation_results: Dictionary containing optimized resource allocations for each institution.
    """

    # Create a linear programming problem
    prob = LpProblem("ResourceAllocationOptimization", LpMaximize)

    # Decision variables
    allocation_vars = {inst: LpVariable(inst, lowBound=0, cat='Integer') for inst in demand_predictions}

    # Objective function: maximize the total demand fulfillment
    prob += lpSum(
        [demand_predictions[inst] * allocation_vars[inst] for inst in demand_predictions]), "TotalDemandFulfillment"

    # Add constraints
    for inst in demand_predictions:
        # Ensure allocated resources do not exceed available resources
        prob += allocation_vars[inst] <= available_resources[inst], f"ResourceConstraint_{inst}"

    # total_allocation_limit
    prob += lpSum([allocation_vars[inst] for inst in demand_predictions]) <= 180, "total_allocation_limit"
    # max_allocation_per_institution
    prob += all(allocation_vars[inst] <= 3 for inst in demand_predictions), "max_allocation_per_institution"
    # min_allocation_per_institution
    prob += all(allocation_vars[inst] >= 1 for inst in demand_predictions), "min_allocation_per_institution"
    # demand_fulfillment_ratio
    prob += lpSum([demand_predictions[inst] * allocation_vars[inst] for inst in demand_predictions]) >= .8 * lpSum(
        [demand_predictions[inst] for inst in demand_predictions]), "demand_fulfillment_ratio"

    # Additional constraints (if any)
    for constraint, value in constraints.items():
        prob += eval(value), f"CustomConstraint_{constraint}"

    # Solve the optimization problem
    prob.solve()

    # Retrieve results
    allocation_results = {inst: allocation_vars[inst].value() for inst in demand_predictions}

    return allocation_results


def get_forcast(institutes, date, test_df):
    demand_forecasts = {}
    for institute in institutes:
        inst_df = test_df[(test_df.index == date) & (test_df.institution == institute)]
        assert inst_df.shape[0] == 1
        pred = XGB_MODEL.predict(inst_df[XGB_COLUMNS])
        demand_forecasts[institute] = int(np.exp(pred)[0])
    return demand_forecasts


if __name__ == '__main__':
    institutions = ['cd059388-fcb3-4c40-b156-2ed1962bd47b',
                    '4bb14745-4254-4247-9b84-bd4fc63583b0',
                    '4a94ba5f-a198-47e0-8eac-817b6991a962',
                    'bd9a1f13-a52c-4372-a63c-1e49bd7a7bf3',
                    '9db090cf-184b-450e-a07a-97dee812ce0d',
                    'f0440cc7-e404-4a2c-bb0d-baf908e9ff25',
                    '60a3c3ea-eff0-4b6a-9d8c-52a6b4fc54e7',
                    'c7537d41-7f0a-4ece-b5bf-4444515b8711',
                    '39f04721-9871-45a8-bc2a-7c892487ba87',
                    'd8c45b45-c70a-47d6-81ba-da8e0cfac6e0',
                    'bf2cf4ca-dac3-4515-ad33-ef66efebeab5',
                    '355cd89d-faa1-4620-8857-a05a24b513e3',
                    '1fb76d53-e58e-4e10-b5ac-1b288a87efb2',
                    '706652e6-6eed-4271-afe8-be8e8322e731',
                    '46412e2a-6b5f-4d0c-9562-d89c01d15259',
                    '7f71654b-0641-4351-897e-60869c7b9de5',
                    'e2adc287-86ec-4341-b43d-13cb91255662',
                    '9e1484a3-0235-48c5-aeed-c44451c918fa',
                    'cf3a43b2-0fef-4db8-873b-6b24e49cfbf3',
                    '7b9c90a6-86d3-487f-971a-cc8e2d9367be',
                    'b05bec66-9c3b-4381-9ad2-0636481ca919',
                    '0da8e9a5-855d-42c5-9435-d69e1019cafc',
                    '49b42879-788e-4347-85cf-23b35a68e746',
                    '4850e9c9-2945-4087-b4f9-6c302e44b84e',
                    '95541ac0-e5ce-493f-b6a4-66142989ad8e',
                    '4389a67a-e4bb-4b93-918b-d417078736ae',
                    'ece0641a-2b44-4cb1-8e31-074dd5fbb4f1',
                    'e81b1231-55bc-488a-abf3-9bf0883430ba',
                    'c7d5aa80-47af-45fd-95d3-c575a44806e5',
                    '99fdda67-36ba-4b00-af8f-0d612fa1ce97',
                    'd21ae02c-0851-4acb-99e9-083d1eeb3f0b',
                    '9db63f62-5c58-4303-919e-e23681909271',
                    '658e24af-453e-4824-a049-5346a5e25964',
                    '8295dc03-9ec7-4f44-9bab-b1ea94e2eef4',
                    '54a4df3e-b280-4c89-9a72-aad7f18b761b',
                    '4a5f0383-2d51-4004-97a6-5b2df37fedd0']

    testing_df = pd.read_csv('data/testing_data.csv', index_col=0)

    demand_predictions = get_forcast(institutions, '2021-07-18', testing_df)
    print("Demand Predictions: ", demand_predictions)
    print()

    # Available resources (number of beds)
    available_resources = {inst: random.randint(1, 7) for inst in institutions}
    print("Available Resources: ", available_resources)
    print()

    # Constraints
    constraints = {
        'inst1': 'allocation_vars["bd9a1f13-a52c-4372-a63c-1e49bd7a7bf3"] + allocation_vars["9db090cf-184b-450e-a07a-97dee812ce0d"] <= 8',
        # Ensure a minimum allocation for specific institutions
        'GeographicalConstraint': 'allocation_vars["9db090cf-184b-450e-a07a-97dee812ce0d"] >= 4',
        'GeographicalConstraint2': 'allocation_vars["bd9a1f13-a52c-4372-a63c-1e49bd7a7bf3"] >= 3',
    }

    # Call the function
    allocation_results = optimize_resource_allocation(demand_predictions, available_resources, constraints)

    # Print the results
    print("Optimized Resource Allocations:")
    for inst, allocation in allocation_results.items():
        print(f"Institution {inst}: {allocation} resources")
