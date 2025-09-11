from fastapi import FastAPI
import uvicorn
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
from pathlib import Path
from pydantic import BaseModel
from scripts.data_cleaning_utils import perform_data_cleaning
from sklearn import set_config

set_config(transform_output='pandas')

class PredictionInput(BaseModel):
    # define the input parameters required for making predictions
    CLIENTNUM: int
    Customer_Age: int  
    Gender: str 
    Dependent_count: int 
    Education_Level: str 
    Marital_Status:  str 
    Income_Category: str 
    Card_Category: str 
    Months_on_book: int  
    Total_Relationship_Count: int  
    Months_Inactive_12_mon: int 
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: int
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

def load_model(model_path):
    model = joblib.load(model_path)
    return model


# mention the columns like this
CATEGORICAL_NOMINAL = ['Gender', 'Marital_Status', 'Income_Category', 'Card_Category']
CATEGORICAL_ORDINAL = ['Education_Level']
NUMERICAL_CONTINUOUS = ['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 
                       'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                       'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio','Avg_Transaction_Value']
NUMERICAL_IMPUTE_MEDIAN = ['Avg_Transaction_Value']
CATEGORICAL_IMPUTE_MODE = ['Education_Level', 'Marital_Status', 'Income_Category']

# root path
root_path = Path(__file__).parent

# model and preprocessor path
model_path = root_path / "models" / "model.joblib"
preprocessor_path = root_path / "models" / "preprocessor.joblib"

# load model and preprocessor
model = load_model(model_path)
preprocessor = load_transformer(preprocessor_path)


model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ('model',model)
])


# create the app
app = FastAPI()


# create the home endpoint
@app.get('/')
def home():
    return "Welcome to credit card churn prediction app."

# create the predict endpoint
@app.post('/predict')
def do_prediction(input_data: PredictionInput):
    pred_data = pd.DataFrame(
        {
            'CLIENTNUM': input_data.CLIENTNUM,
            'Customer_Age': input_data.Customer_Age,
            'Gender': input_data.Gender,
            'Dependent_count': input_data.Dependent_count,
            'Education_Level': input_data.Education_Level,
            'Marital_Status': input_data.Marital_Status,
            'Income_Category': input_data.Income_Category,
            'Card_Category': input_data.Card_Category,
            'Months_on_book': input_data.Months_on_book,
            'Total_Relationship_Count': input_data.Total_Relationship_Count,
            'Months_Inactive_12_mon': input_data.Months_Inactive_12_mon,
            'Contacts_Count_12_mon': input_data.Contacts_Count_12_mon,
            'Credit_Limit': input_data.Credit_Limit,
            'Total_Revolving_Bal': input_data.Total_Revolving_Bal,
            'Avg_Open_To_Buy': input_data.Avg_Open_To_Buy,
            'Total_Amt_Chng_Q4_Q1': input_data.Total_Amt_Chng_Q4_Q1,
            'Total_Trans_Amt': input_data.Total_Trans_Amt,
            'Total_Trans_Ct': input_data.Total_Trans_Ct,
            'Total_Ct_Chng_Q4_Q1': input_data.Total_Ct_Chng_Q4_Q1,
            'Avg_Utilization_Ratio': input_data.Avg_Utilization_Ratio
        },index=[0]
    )

    # clean the input data
    df_cleaned = perform_data_cleaning(pred_data)


    # get the predictions
    prediction = model_pipe.predict(df_cleaned)[0]
    probability = model_pipe.predict_proba(df_cleaned)[0][1]

    label_map = {0: "Existing Customer", 1: "Attrited Customer"}
    label = label_map[prediction]

    return {
        "prediction": int(prediction),        
        "label": label,  
        "churn_probability": float(probability)
    }


if __name__ == "__main__":

    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)

