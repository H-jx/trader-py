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
from trader.Backtest import Backtest
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
    df = pd.read_csv('./data/sample-data.csv')
    df = df.dropna()  # 删除包含 NaN 的行
    df['sclose'] = df['close']
    df['long_rsi'] = df['long_rsi'] / 100
    keys = [
        'sclose',
        'close_less_than_ma10',
        'ma10_less_than_ma30',
        'ma30_less_than_ma60',
        'sell_rate',
        'low_boll_rate',
        'high_boll_rate',
        'boll_range_rate',
        'changepercent',
        'upper_shadow_rate',
        'lower_shadow_rate',
        'close_ma60_rate',
        'volume_ma20_rate',
        'long_rsi'
    ]
    df[keys + ['close']] = df[keys + ['close']].astype(float)

    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    normalize(df, [
        'sclose'
    ])
   
    
    print(df.head(20))
    backtest = Backtest(trade_volume = 0.4, balance= 1600, position = 0)
    # 创建TradingEnv实例
    env = TradingEnv(df = df, keys=keys, backtest=backtest)

    # 定义模型和超参数
    model = DQN("MlpPolicy", env, learning_rate=1e-5, buffer_size=100000, batch_size=32, verbose=0, device='cuda')
    # env.load_model('./modes/mode.zip')
    # model = ACER("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    # df数据长度

    # 开始训练数据
    model.learn(total_timesteps=len(df) * 10, tb_log_name='run')

    # 回测
    obs = env.reset()

    for i in range(len(df) - 1):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render(action)
        if done:
            break

    if env.get_profit() > 0:
        model.save('./modes/DQN')

    print('Profit: %.2f%%' % (env.get_profit()))

    with open('./data/predict.json', 'w') as f:
        json.dump(env.backtest.trades, f)

 
# 标准化函数
def normalize(df, cols):
    """
    对 DataFrame 中的指定列进行标准化
    
    Args:
        df (pandas.DataFrame): 要标准化的 DataFrame
        cols (List[str]): 需要标准化的列名列表
        
    Returns:
        pandas.DataFrame: 标准化后的 DataFrame
    """
    # 按列计算平均值和标准差
    means = df[cols].mean()
    stds = df[cols].std()

    # 标准化
    df[cols] = (df[cols] - means) / stds

    return df