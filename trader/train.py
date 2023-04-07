import json
import logging
import csv
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from stable_baselines3.dqn.dqn import DQN

from trader.TradingEnv import TradingEnv
from trader.util import get_interval

# from stable_baselines import ACER
# from collections import deque
import tensorflow as tf
import torch

# 用此方法检查，有效。
torch.zeros(1).cuda()
# print(tf.test.is_built_with_cuda())

def get_history(symbol: str, start_time: str='', end_time: str=''):
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
        return ''

def download():
    start_time, end_time = ["2023/01/08 18:26:44", "2023/03/15 18:26:44"]
    symbol = "ETHUSDT"
    print(start_time, end_time)
    csvStr = get_history(symbol, start_time, end_time)
    # 将数据写入JSON文件
    # with open(f"data/{symbol}-{start_time[0:10].replace('/', '-')}.csv", 'w') as f:
    #     json.dump(csvStr, f)

    with open(f"data/{symbol}-{start_time[0:10].replace('/', '-')}.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in csvStr.split('\n'):
            writer.writerow(row.split(','))

download()

def analyze_data():
    
    # 读取JSON文件
    with open('./data/ETHUSDT-2023-01-08.json', 'r') as f:
        data = json.load(f)
    # 转换为DataFrame
    df = pd.DataFrame(data)
    # 计算5日均线和10日均线

    df['close'] = pd.to_numeric(df['usdtPrice'])
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['close/ma60'] = (df['close'] - df['ma60']) / df['ma60']
    df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
    df['sell'] = pd.to_numeric(df['sell'], errors='coerce')
    df['amount'] = df['buy'] + df['sell']
    df['buy/amount'] = df['buy'] / df['amount']
    df['sell/amount'] = df['sell'] / df['amount']
    df['amount_ma20'] = df['amount'].rolling(window=20).mean()
    df['amount/amount_ma20'] = df['amount'] / df['amount_ma20']
    
    # df['amount/amount_ma20'] = df['amount/amount_ma20'].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else x))
    # time的格式如下：2023-03-14T15:40:00.000Z， 我想转为时间戳
    df['timestamp'] = pd.to_datetime(df['time'], errors='coerce').apply(lambda x: x.timestamp())
    df['changepercent'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df = df.dropna().reset_index(drop=True)
    # df['label'] = 0
    # print(df['timestamp'].head(5))
    # df.loc[(df['changepercent'] > 0.1) & (df['amount_ma20'] > 1) & df['close/ma60'] > 0, 'label'] = 1
    # df.loc[(df['changepercent'] < -0.1) & (df['amount_ma20'] > 1) & df['close/ma60'] < 0, 'label'] = -1

    print(df.head(5))


    scaler = StandardScaler()

    df2 = pd.DataFrame()
    
  
   
    df2['sclose'] = df['close']
    # df2['ma30'] = df['ma30']
    df2['ma60'] = df['ma60']
    # 标准化处理
    scaler.fit(df2)
    df2 = pd.DataFrame(scaler.transform(df2), columns=df2.columns)
    df2['amount/amount_ma20'] = df['amount/amount_ma20']
    df2['close/ma60'] = df['close/ma60']
    df2['time'] = df['time']
    df2['timestamp'] = df['timestamp']
    df2['close'] = df['close']
    df2['buy/amount'] = df['buy/amount']
    df2['sell/amount'] = df['sell/amount']
    df2['changepercent'] = df['changepercent']
    print(df2.head(20))
    trades = []

    # 创建TradingEnv实例
    env = TradingEnv(df = df2,  keys=['sclose', 'ma60', 'close/ma60', 'buy/amount', 'sell/amount', 'amount/amount_ma20', 'changepercent'])

    # 定义模型和超参数
    model = DQN("MlpPolicy", env, learning_rate=1e-5, buffer_size=100000, batch_size=128, verbose=0, device='cuda')
    # env.load_model('./modes/mode.zip')
    # model = ACER("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    # df数据长度

    # 开始训练数据
    model.learn(total_timesteps=16165 * 50, tb_log_name='run')

    # 回测
    obs = env.reset()

    for i in range(len(df) - 1):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render(action)
        trades.append(info)
        if done:
            break
    if env.get_profit() > 0:
        model.save('./modes/mode2')

    print('Profit: %.2f%%' % (env.get_profit() * 100))

    with open('./data/predict.json', 'w') as f:
        json.dump(trades, f)

 
