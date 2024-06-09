from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def evaluate():
    return {"message": "Hello World"}
