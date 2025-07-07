import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
from data_function import *

def get_dataset():
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA", "KRW-BCH", "KRW-ETC", "KRW-LINK", "KRW-ATOM", "KRW-CHZ", "KRW-AXS", "KRW-ALGO", "KRW-STX", "KRW-IOST"]
    all_data = pd.DataFrame()
    pd.set_option('display.max_columns', None)
    for market in markets:
        df = data_load(market, count=250)
        df["coin"] = market  # coin 컬럼 추가
        all_data = pd.concat([all_data, df])
        time.sleep(0.2)

    print(all_data.head())
    all_data = all_data.sort_values(by=["coin", "date"]).reset_index(drop=True)

    all_data["label"] = all_data["close"].shift(-1)
    all_data["label"] = (all_data["label"] > all_data["close"]).astype(int)
    all_data = all_data.dropna().reset_index(drop=True)

    x_train, y_train = create_sequences(all_data, window_size=14)
    x_train = data_normalization(x_train, num_feature_idx=[0, 1, 2, 3, 4])


    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    test_market=["KRW-DOGE", "KRW-BORA", "KRW-AVAX", "KRW-SOL"]
    df_test=pd.DataFrame()
    for market in test_market:
        df = data_load(market, count=250)
        df["coin"] = market  # coin 컬럼 추가
        df_test = pd.concat([df_test, df])
        time.sleep(0.2)
    df_test = df_test.sort_values(by=["coin", "date"]).reset_index(drop=True)
    df_test["label"] = df_test["close"].shift(-1)
    df_test["label"] = (df_test["label"] > df_test["close"]).astype(int)
    df_test = df_test.dropna().reset_index(drop=True)

    x_test, y_test = create_sequences(df_test, window_size=14)
    x_test = data_normalization(x_test, num_feature_idx=[0, 1, 2, 3, 4])
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # validation
    val_market=["KRW-ETC", "KRW-XLM", "KRW-LINK", "KRW-UNI"]
    df_val=pd.DataFrame()
    for market in val_market:
        df = data_load(market, count=250)
        df["coin"] = market  # coin 컬럼 추가
        df_val= pd.concat([df_val, df])
        time.sleep(0.2)
    df_val = df_val.sort_values(by=["coin", "date"]).reset_index(drop=True)
    df_val["label"] = df_val["close"].shift(-1)
    df_val["label"] = (df_val["label"] > df_val["close"]).astype(int)
    df_val = df_val.dropna().reset_index(drop=True)

    x_val, y_val = create_sequences(df_val, window_size=14)
    x_val = data_normalization(x_val, num_feature_idx=[0, 1, 2, 3, 4])
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

    return train_loader, val_loader, x_test, y_test
