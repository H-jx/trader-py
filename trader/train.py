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

    keys = [
        'timestamp',
        'symbol',
        'close',
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
    
    print(df.head(20))
    backtest = Backtest(trade_volume = 0.1, balance= 1400, volume = 0.1)
    # 创建TradingEnv实例
    env = TradingEnv(df = df, keys=keys, backtest=backtest)

    # 定义模型和超参数
    model = DQN("MlpPolicy", env, learning_rate=1e-5, buffer_size=100000, batch_size=128, verbose=0, device='cuda')
    # env.load_model('./modes/mode.zip')
    # model = ACER("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    # df数据长度

    # 开始训练数据
    model.learn(total_timesteps=16165 * 50, tb_log_name='run')

    # 回测
    obs = env.reset()
    trades = []
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

 
