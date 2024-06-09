import mlflow
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def evaluate():
    model = mlflow.pyfunc.load_model("./model")
    print(type(model))
    return {"message": "Hello World"}
