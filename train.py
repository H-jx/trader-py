import sys
import os
import json
import requests
import logging
import pandas as pd
from util import get_interval

def get_history(symbol: str, start_time: str=None, end_time: str=None):
    """Fetches trade history for a given symbol and time range.

    Args:
        symbol: str
        start_time: str
        end_time: str

    Returns: List[Dict]

    Raises:
        HTTPError: If the API returns an error response.
    """                     
    # 读取接口数据
    url = 'http://trader.8and1.cn/api/trade-history'

    params = {'start': start_time, 'end': end_time, 'symbol': symbol}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data['data']
    except Exception as e:
        logging.error(f"{url}: {e}")

def start():
    start_time, end_time = get_interval()
    symbol = "ETHUSDT"
    arr = get_history(symbol, start_time, end_time)
    # 将数据写入JSON文件
    with open(f"data/{symbol}-{start_time[0:10]}.json", 'w') as f:
        json.dump(arr, f)

# start()

def analyze_data():
    # 读取JSON文件
    with open('data/ETHUSDT-2022-12-14.json', 'r') as f:
        data = json.load(f)
    # 转换为DataFrame
    df = pd.DataFrame(data)
    # 计算5日均线和10日均线
    df['MA5'] = df['usdtPrice'].rolling(window=5).mean()
    df['MA10'] = df['usdtPrice'].rolling(window=10).mean()
    df['MA30'] = df['usdtPrice'].rolling(window=30).mean()
    df['MA60'] = df['usdtPrice'].rolling(window=60).mean()
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['sell'] = pd.to_numeric(df['sell'], errors='coerce')
    df['amount'] = df['buy'] + df['sell']
    df['amountMA20'] = df['amount'].rolling(window=20).mean()
    print(df.head(5))

analyze_data()