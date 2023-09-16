# -*- coding:utf-8 -*-  
import json
import logging
import csv
import datetime
import pandas as pd
import requests
import os
# from sklearn.preprocessing import StandardScaler
from stable_baselines3.dqn.dqn import DQN

from trader.TradingEnv import TradingEnv
from trader.util import get_interval, compress_data
from trader.Backtest import Backtest
# from stable_baselines import ACER
# from collections import deque
import tensorflow as tf
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 用此方法检查，有效。
# torch.zeros(1).cuda()
# print(tf.test.is_built_with_cuda())

def get_history(symbol: str, start_time: int, end_time: int):
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
    url = 'http://trader.8and1.cn/api/kline-history'

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
    start_time, end_time = ["2020/8/1 00:00:00", "2023/04/4 00:00:00"]
    symbol = "ETHUSDT"
    print(start_time, end_time)

    # start_time, end_time使用时间戳
    start_time = int(datetime.datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S").timestamp() * 1000)
    end_time = int(datetime.datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S").timestamp() * 1000)
    csvStr = get_history(symbol, start_time, end_time)
    # 将双引号替换为空
    csvStr = csvStr.replace('\\"', '')
    # 将数据写入JSON文件
    # with open(f"data/{symbol}-{start_time[0:10].replace('/', '-')}.csv", 'w') as f:
    #     json.dump(csvStr, f)

    with open(f"data/{symbol}-2020.csv", 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in csvStr.split('\n'):
            writer.writerow(row.split(','))

# download()
def get_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna()  # 删除包含 NaN 的行
    df['rsi_4h'] = df['rsi_4h'] / 100
    keys = [
        # 'close_less_than_ma10_5m',
        # 'ma10_5m_less_than_ma30_5m',
        # 'ma30_5m_less_than_ma60_5m',
        # 'change_rate_5m',
        # 'boll_range_rate_5m',
        'close_ma60_rate_5m',
        'close_ma60_rate_4h',
        'volume_ma20_rate_5m',
        'volume_ma20_rate_4h',
        'rsi_4h',
        'close_ma30_rate_1d',
    ]
    df[keys + ['close', 'open', 'high']] = df[keys + ['close', 'open', 'high']].astype(float)

    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

    compress_data(df, ['volume_ma20_rate_5m', 'volume_ma20_rate_4h'])

    return [df.head(int(len(df)*0.5)), keys]

def analyze_data():
    df, keys  = get_data('./data/sample-data.csv')
   
    print(df.head(20))
    backtest = Backtest(trade_volume = 0.1, balance= 4600, position = 0)


    # 创建TradingEnv实例
    env = TradingEnv(df = df, keys=keys, backtest=backtest)
    # env.load_model('./modes/DQN.zip')

    loaded_model = DQN.load('./models/DQN')
    env.model = loaded_model
    # 定义模型和超参数
    model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=100000, batch_size=64, verbose=0, device='cuda')
   
    # model = ACER("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    # df数据长度

    # 开始训练数据
    model.learn(total_timesteps=len(df) * 10, tb_log_name='run')

    # 回测
    df, keys  = get_data('./data/predict-data.csv')
    env.set_df(df)
    obs = env.reset()
    print('predict')
    
    for i in range(len(df) - 1):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    if env.get_profit() > 0:
        model.save('./models/DQN')
    env.close()
    print('Profit: %.2f%%' % (env.get_profit() * 100))
    print(env.backtest.get_results())

    with open('./data/predict.json', 'w') as f:
        json.dump(env.backtest.get_trades(), f)

 
def train2():

    df, keys = get_data('./data/sample-data.csv') 
    fig, ax = plt.subplots(figsize=(15, 5))

    # 处理数据
    X = df[keys].values
    X = X.reshape((-1, 1, 1, len(keys), 1))
    y = df['close'].values

    # 拆分数据集
    split_idx = int(0.8*len(df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 定义ConvLSTM模型
    model = keras.models.Sequential()
    model.add(keras.layers.ConvLSTM2D(32, (1, len(keys)), input_shape=(1, 1, len(keys), 1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mae') 

    # 训练模型
    model.fit(X_train, y_train, 
            epochs=20, 
            validation_data=(X_test, y_test))

    # 预测并计算MAE
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))

    ax.plot(y_test, color='b', label='True')
    ax.plot(y_pred, color='r', label='Predicted')
    ax.set_xlabel('Time')  
    ax.set_ylabel('Close Price')
    ax.legend()
    print('MAE:', mae)
    plt.show()
    plt.close('all')