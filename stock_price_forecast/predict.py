import argparse
import torch
from .model import GRU
from .data_function import data_load, create_sequences, data_normalization
import numpy as np
import time

def predict(market, model_path, window_size=14, num_features=9):
    df = data_load(market, count=window_size)
    df["label"] = df["close"].shift(-1)
    df["label"] = (df["label"] > df["close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    # x_seq, _ = create_sequences(df, window_size=3)
    feature_columns = ["open", "high", "low", "change", "volume",
                       "day_sin", "day_cos", "month_sin", "month_cos"]
    x_seq = df[feature_columns].iloc[0: window_size].values
    x=[x_seq]
    x_seq=np.array(x)
    x_seq = data_normalization(x_seq, num_feature_idx=[0, 1, 2, 3, 4])
    x_tensor = torch.tensor(x_seq[-1:], dtype=torch.float32)
    model = GRU(input_size=num_features, hidden_size=64, num_layers=2, seq_length=window_size, num_classes=1)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(x_tensor).item()
        prediction = True if output >= 0.5 else False

    if prediction == True:
        print("상승")
    else:
        print("하락")
    
    return prediction

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, required=True, help="코인 마켓 코드 (예: KRW-BTC)")
    parser.add_argument("--model_path", type=str, default="C:/Users/User/PycharmProjects/PythonProject/stock_price_forecast/best_model.pth", help="학습된 모델 경로")
    args = parser.parse_args()

    predict(args.market, args.model_path)

    end_time = time.time()
    print(f"\n✅ 실행 시간: {end_time - start_time:.2f}초")