import os 
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(nums_attribs,cat_attribs):
    #num pipeline 
    num_pipeline = Pipeline ([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    #cat pipeline
    cat_pipeline = Pipeline([
        ("Encoder",OneHotEncoder())
    ])

    #full Pipeline
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,nums_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    
    #load csv
    housing = pd.read_csv("housing.csv")
    housing = pd.DataFrame(housing)

    #test and train split
    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat",axis =1)
        test_set = housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False)
    
    housing = train_set.copy()

    #sepatrate the data from labels 
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value",axis=1)

    #list the num and cat attribs
    num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    #pipeline building 
    pipeline  = build_pipeline(num_attribs,cat_attribs)

    model = RandomForestRegressor()
    housing_prepared = pipeline.fit_transform(housing)
    model.fit(housing_prepared,housing_labels)

    #save the pipeline and model
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model Trained Sucessfully and save !!")

else:
    #loadd the model and pipeline 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["predictions"] = predictions
    input_data.to_csv("output.csv",index=False)
    print("Interface is comepleted , the predictions are saved into the output file sucessfully !!")