import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib.dates as mdates
def train(model, x_train, y_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch + 1, "Loss:", loss.item())

def test(model, x_test, y_test, criterion, df):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test)
        predicted = outputs.detach().cpu().numpy()
        label_y = y_test.detach().numpy()
        ms=MinMaxScaler()
        ms.fit(predicted)
        predicted = ms.inverse_transform(predicted)
        label_y = ms.inverse_transform(label_y)
        plt.figure(figsize=(10, 6))
        plt.axvline(x=mdates.date2num(datetime(2024, 4, 1)), color='r', linestyle='--')
        df['A'] = outputs[:len(df)]
        # df['pred']=outputs[]
        plt.plot(df['Close'], label='Actual Data')
        plt.plot(df['pred'], label='Predicted Data')

        plt.title('Time-series Prediction')
        plt.legend()
        plt.show()

        mse = criterion(label_y, predicted)
        r2 = r2_score(label_y, predicted)

        # 성능 지표 출력
        print("Model Performance:")
        print(f" - Mean Squared Error (MSE): {mse:.4f}")
        print(f" - R-squared (R²): {r2:.4f}")