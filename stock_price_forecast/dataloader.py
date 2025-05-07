import requests
import pandas as pd
import time
from data_function import *
import torch


def get_dataset():
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]

    all_data = pd.DataFrame()

    for market in markets:
        df = data_load(market, count=120)
        all_data = pd.concat([all_data, df])
        time.sleep(0.2)

    all_data = all_data.sort_values(by=["coin", "date"]).reset_index(drop=True)
    onehot = pd.get_dummies(all_data["coin"])
    all_data = pd.concat([all_data, onehot], axis=1)
    datas, labels = create_sequences(all_data, window_size=3)
    datas = data_normalization(datas, num_feature=[0, 1, 2, 3, 4])

    x_train, y_train, x_val, y_val = split_data(datas, labels, train_size=0.8, val_size=0.2)
    x_train = x_train.astype(float)
    x_val = x_val.astype(float)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Test Data
    df = data_load("KRW-DOGE", count=120)
    x_test = []
    y_test = []
    feature_columns = ['open', 'high', 'low', 'change', 'volume']
    for i in range(len(df) - 3):
        seq = df[feature_columns].iloc[i:i + 3].values
        label = df.iloc[i + 3]['close']

        x_test.append(seq)
        y_test.append(label)

    x_test = np.array(x_test)
    x_test = data_normalization(x_test, num_feature=[0, 1, 2, 3, 4])
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test