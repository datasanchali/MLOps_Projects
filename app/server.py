from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("app/model.joblib")

class_names = np.array(["setosa", "versicolor", "virginica"])

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}


@app.post("/predict")
def predict(data: dict):
    """
    Predict the class of iris flower based on input features.
    Args:
        data (dict): A dictionary containing the features of the iris flower.
        eg: {"features": [5.1, 3.5, 1.4, 0.2]}
    Returns:
        dict: A dictionary containing the predicted class of the iris flower.
    """
    # features
    features = np.array(data["features"]).reshape(1, -1)
    # make a prediction
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {"predicted_class": class_name}
