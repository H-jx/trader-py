import sys
import os
import json
import requests
import logging
import pandas as pd
from stable_baselines3 import DQN
from collections import deque
from util import get_interval
from TradingEnv import TradingEnv
from sklearn.preprocessing import StandardScaler

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
        # if data['message']:
        #     print(data['message'])
        #     return
        return data['data']
    except Exception as e:
        logging.error(f"{url}: {e}")

def download():
    start_time, end_time = get_interval(1200)
    symbol = "ETHUSDT"
    print(start_time, end_time)
    arr = get_history(symbol, start_time, end_time)
    # 将数据写入JSON文件
    with open(f"data/{symbol}-{start_time[0:10]}.json", 'w') as f:
        json.dump(arr, f)

# download()

def analyze_data():
    # 读取JSON文件
    with open('data/ETHUSDT-2023-01-08.json', 'r') as f:
        data = json.load(f)
    # 转换为DataFrame
    df = pd.DataFrame(data)
    # 计算5日均线和10日均线

    df['close'] = pd.to_numeric(df['usdtPrice'], errors='coerce')

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['close/ma60'] = (df['close'] - df['ma60']) / df['ma60']
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['sell'] = pd.to_numeric(df['sell'], errors='coerce')
    df['amount'] = df['buy'] + df['sell']
    df['buy/amount'] = df['buy'] / df['amount']
    df['sell/amount'] = df['sell'] / df['amount']
    df['amount_ma20'] = df['amount'].rolling(window=20).mean()
    df['amount/amount_ma20'] = df['amount'] / df['amount_ma20']
    df['amount/amount_ma20'] = df['amount/amount_ma20'].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else x))
    df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
    df['changepercent'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['label'] = 0

    df.loc[(df['changepercent'] > 0.1) & (df['amount_ma20'] > 2), 'label'] = 1
    df.loc[(df['changepercent'] < -0.1) & (df['amount_ma20'] > 2), 'label'] = -1



    print(df.head(5))

    # df.to_excel('data/out.xlsx', index=False)
    df.dropna(inplace=True)
    scaler = StandardScaler()
    # 标准化处理
    df2 = pd.DataFrame()
    df2['changepercent'] = df['changepercent']
    # df2['close'] = df['close']
    # df2['ma60'] = df['ma60']
    scaler.fit(df2)
    df2 = pd.DataFrame(scaler.transform(df2), columns=df2.columns)
    df2['close/ma60'] = df['close/ma60']
    df2['buy/amount'] = df['buy/amount']
    df2['sell/amount'] = df['sell/amount']
    df2['amount/amount_ma20'] = df['amount/amount_ma20']
    print(df2.tail(5))
    # 创建TradingEnv实例
    env = TradingEnv(df)

    # 定义模型和超参数
    model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000, batch_size=64, verbose=0)
    model.learn(total_timesteps=100000)

    # 回测
    obs = env.reset()
    for i in range(len(df) - 1):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    print('Profit: %.2f%%' % (env.profit * 100))
    env.save_model('./mode', model)
analyze_data()
 
