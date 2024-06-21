import json

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class MovieInput(BaseModel):
    userId: list[int]
    movieId: list[int]


app = FastAPI()


@app.post("/predict")
async def evaluate(movie_input: MovieInput):
    movie_input_dict = movie_input.model_dump()
    df_input = pd.DataFrame(movie_input_dict)

    model = mlflow.pyfunc.load_model("./model")
    preprocessor = mlflow.pyfunc.load_model("./preprocessor")
    df_input: pd.DataFrame = preprocessor.predict(df_input)
    predictions_np: np.ndarray = model.predict(df_input).detach().numpy()
    reshaped = predictions_np.reshape(1, -1).tolist()[0]
    return {"predictions": json.dumps(reshaped)}


@app.get("/metrics")
async def get_metrics():
    with open("./metrics.json", "r") as fp:
        metrics_dict = json.load(fp)
    return metrics_dict
