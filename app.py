"""HR attrition API. The notebook saves the model and its settings;
this file serves them. Run it with: uvicorn app:app --reload
"""
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

model = joblib.load('hr_attrition_model.joblib')
with open('model_meta.json') as handle:
    meta = json.load(handle)

THRESHOLD = meta['threshold']
FEATURES = meta['features']

# Build the input schema from the features the model was trained on, so the
# API always matches the model, whichever features you removed.
EmployeeRecord = create_model(
    'EmployeeRecord', **{name: (float, ...) for name in FEATURES})

app = FastAPI(title='HR Attrition Classifier', version='1.0')


@app.get('/health')
def health():
    return {'status': 'ok', 'threshold': THRESHOLD, 'features': len(FEATURES)}


@app.post('/predict')
def predict(employee: EmployeeRecord):
    data = pd.DataFrame([employee.model_dump()])[FEATURES]
    probability = float(model.predict_proba(data)[0, 1])
    return {
        'attrition_probability': round(probability, 4),
        'flag': int(probability >= THRESHOLD),
        'threshold_used': THRESHOLD,
    }
