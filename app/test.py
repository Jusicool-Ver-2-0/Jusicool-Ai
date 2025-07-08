import requests

# FastAPI 서버 주소
url = "http://127.0.0.1:8000/predict/"

data = {
    "market": "KRW-BTC",
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
