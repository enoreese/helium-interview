from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

DB = pd.read_csv('data/testing_data.csv', index_col=0)

XGB_MODEL = joblib.load('models/xgboost_t1_pipeline.sav')
XGB_COLUMNS = ['institution', 'inst_type', 'month', 'dayofweek',
               'visit_count', 'no_unique_patients', 'no_out_patients',
               'no_in_patients', 'in_out_ratio', 'avg_age', 'avg_male_age',
               'avg_female_age', 'max_age', 'min_age', 'no_male',
               'no_female', 'no_unique_states', 'day', 'lag_1', 'lag_2', 'lag_3']


def get_forcast(institute, date):
    inst_df = DB[(DB.index == date) & (DB.institution == institute)]
    if inst_df.shape[0] == 0:
        return None

    pred = XGB_MODEL.predict(inst_df[XGB_COLUMNS])
    demand_forecast = int(np.exp(pred)[0])
    return demand_forecast


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/demand/")
def forecast_demand(institution: str, date: str):
    if not institution:
        return {
            "status": "error", "message": "No institution passed"
        }
    if not date:
        return {
            "status": "error", "message": "No date passed"
        }

    date = pd.to_datetime(date)

    # Ideally, run some data transformations

    demand_pred = get_forcast(institution, date)
    if not demand_pred:
        return {
            "status": "error", "message": "No DB entry found"
        }

    return {
        "status": "error", "institution": institution, "demand_forecast": demand_pred
    }
