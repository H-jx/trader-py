# -*- coding:utf-8 -*-  
import json
import logging
import csv
import datetime
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

def analyze_data():

    # 转换为DataFrame
    df = pd.read_csv('./data/ETHUSDT-2020.csv')
    # df = df[df['symbol'] == 'ETHUSDT']

    # 计算5日均线和10日均线
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['close_ma60_rate'] = (df['close'] - df['ma60']) / df['ma60']
    df['buy_rate'] = df['buy'] / df['volume']
    df['sell_rate'] = df['sell'] / df['volume']
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_volume_ma20'] = df['volume'] / df['volume_ma20']

    # time的格式如下：2020-08-01T16:00:00.000Z,， 我想转为时间戳
    df['time'] = pd.to_datetime(df['time'])
    df['changepercent'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)


    # buy_filtered = df.loc[df['label'] == 1]

    # close_ma60_range = (buy_filtered['close_ma60'].min(), buy_filtered['close_ma60'].max())
    # buy_rate_range = (buy_filtered['sell_rate'].min(), buy_filtered['sell_rate'].max())
    # volume_volume_ma20_range = (buy_filtered['volume_volume_ma20'].min(), buy_filtered['volume_volume_ma20'].max())

    # print('buy')
    # print('close_ma60_range: ', close_ma60_range)
    # print('sell_rate_range: ', buy_rate_range)
    # print('volume_volume_ma20_range: ', volume_volume_ma20_range)
    # print(buy_filtered.loc[buy_filtered['close_ma60'] > 0.01])

    # sell_filtered = df.loc[df['label'] == -1]
    # close_ma60_range = (sell_filtered['close_ma60'].min(), sell_filtered['close_ma60'].max())
    # sell_rate_range = (sell_filtered['buy_rate'].min(), sell_filtered['buy_rate'].max())
    # volume_volume_ma20_range = (sell_filtered['volume_volume_ma20'].min(), sell_filtered['volume_volume_ma20'].max())

    # print('-----')
    # print('sell')
    # print('close_ma60_range: ', close_ma60_range)
    # print('sell_rate_range: ', sell_rate_range)
    # print('volume_volume_ma20_range: ', volume_volume_ma20_range)
    # print(sell_filtered.head(20))

    scaler = StandardScaler()

    df2 = pd.DataFrame()
    
    
   
    df2['sclose'] = df['close']
    df2['ma30'] = df['ma30']
    df2['ma60'] = df['ma60']
    df2['close_ma60_rate'] = df['close_ma60_rate']
    df2['buy_rate'] = df['buy_rate']
    df2['sell_rate'] = df['sell_rate']
    df2['volume_volume_ma20'] = df['volume_volume_ma20']
    df2['changepercent'] = df['changepercent']
    # 标准化处理
    scaler.fit(df2)
    df2 = pd.DataFrame(scaler.transform(df2), columns=df2.columns)
    df2['time'] = df['time']
    df2['timestamp'] = df['timestamp']
    df2['close'] = df['close']
    print(df2.head(20))
    trades = []

    # 创建TradingEnv实例
    env = TradingEnv(df = df2,  keys=['sclose', 'ma30', 'ma60', 'close_ma60_rate', 'buy_rate', 'sell_rate', 'volume_volume_ma20', 'changepercent', 'timestamp'])

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

 
