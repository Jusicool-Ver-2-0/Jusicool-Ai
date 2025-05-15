import requests
import pandas as pd
import numpy as np

def data_load(market, count=120):
    url = f"https://api.upbit.com/v1/candles/days?market={market}&count={count}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()

    df = pd.DataFrame(data)
    df = df[["candle_date_time_kst", "opening_price", "high_price", "low_price",
             "trade_price", "candle_acc_trade_volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["change"] = df["close"].pct_change().fillna(0)
    df["day_sin"] = np.sin(2 * np.pi * df["date"].dt.day / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["date"].dt.day / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    return df

def create_sequences(df, window_size=3):
    feature_columns = ["open", "high", "low", "change", "volume",
                       "day_sin", "day_cos", "month_sin", "month_cos"]
    x = []
    y = []
    for i in range(len(df) - window_size):
        seq_x = df[feature_columns].iloc[i:i+window_size].values
        label = df["label"].iloc[i+window_size]
        x.append(seq_x)
        y.append(label)
    return np.array(x), np.array(y)

def data_normalization(x, num_feature_idx):
    for i in num_feature_idx:
        mean = x[:, :, i].mean()
        std = x[:, :, i].std()
        x[:, :, i] = (x[:, :, i] - mean) / std
    return x

def split_data(x, y, train_size=0.8, val_size=0.2):
    train_len = int(len(x) * train_size)

    x_train = x[:train_len]
    y_train = y[:train_len]
    x_val = x[train_len:]
    y_val = y[train_len:]

    return x_train, y_train, x_val, y_val
