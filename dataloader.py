import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
def get_dataset():
    df = yf.download('035420.KS', start='2023-06-01', end='2024-06-30')
    # print(df.columns)
    x=df.drop('Close', axis=1)
    y=df[['Close']]
    # print(x.shape)
    ms=MinMaxScaler()
    ss=StandardScaler()
    x=ss.fit_transform(x)
    y=ms.fit_transform(y)
    x_train=x[:74, :]
    x_test=x[74:, :]
    y_train=y[:74, :]
    y_test=y[74:, :]
    x_train=torch.tensor(x_train, dtype=torch.float32)
    x_test=torch.tensor(x_test, dtype=torch.float32)
    y_train=torch.tensor(y_train, dtype=torch.float32)
    y_test=torch.tensor(y_test, dtype=torch.float32)
    x_train=torch.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test=torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


    return x_train, y_train, x_test, y_test, df