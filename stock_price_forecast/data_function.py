import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def data_load(market, count=120):
    url = "https://api.upbit.com/v1/candles/days"
    params = {
        "market": market,
        "count": count
    }
    headers = {"accept": "application/json"}

    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    df = pd.DataFrame(data)

    df = df.rename(columns={
        "candle_date_time_kst": "date",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_price": "volume"
    })

    df["coin"] = market
    df["date"] = pd.to_datetime(df["date"])
    df["change"] = df["close"].pct_change()* 100
    return df[["coin","date", "open", "high", "low", "close", "volume", "change"]]

def create_sequences(df, window_size=3):
    sequences = []
    labels = []
    feature_columns = ['open', 'high', 'low', 'change', 'volume',
                       "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]

    grouped = {coin: df[df['coin'] == coin].reset_index(drop=True) for coin in df['coin'].unique()}
    coin_names = list(grouped.keys())

    min_len = min([len(data) for data in grouped.values()])

    for i in range(min_len - window_size):
        for coin in coin_names:
            coin_data = grouped[coin]
            seq = coin_data[feature_columns].iloc[i:i + window_size].values
            label = coin_data.iloc[i + window_size]['close']

            sequences.append(seq)

            labels.append(label)

    return np.array(sequences), np.array(labels)

def data_normalization(datas, num_feature):
    datas_copy = datas.copy()
    n_samples, window_size, n_features = datas.shape

    datas_2d = datas[:, :, num_feature].reshape(-1, len(num_feature))

    scaler = MinMaxScaler()
    scaled_2d = scaler.fit_transform(datas_2d)

    datas_copy[:, :, num_feature] = scaled_2d.reshape(n_samples, window_size, len(num_feature))
    return datas_copy

def split_data(datas, labels, train_size=0.8, val_size=0.2):
    data_size= len(datas)
    train_count= int(data_size * train_size)

    x_train= datas[:train_count]
    y_train= labels[:train_count]
    x_val= datas[train_count:]
    y_val= labels[train_count:]
    return x_train, y_train, x_val, y_val