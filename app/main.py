from fastapi import FastAPI
import argparse
import torch
from pydantic import BaseModel
from stock_price_forecast.predict import predict  # Assuming the predict function is in the same directory

app = FastAPI()

class InputData(BaseModel):
    market: str
    model_path: str = "best_model.pth"
    

@app.post("/predict/")
def upbi_predict(data: InputData):
    result = predict(market=data.market, model_path=data.model_path)
    return {"result": result}
