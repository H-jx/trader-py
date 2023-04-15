
import pickle
from typing import List

import gym
from gym import spaces
import numpy as np
import pandas as pd
from trader.Backtest import Backtest

# 用python实现一个TradingEnv(gym.Env)，功能为通过虚拟货币(ETHUSDT)历史数据训练模型
# 1. 训练数据列： 'close' 'ma60' 'close/ma60' 'buy/amount' 'sell/amount' 'amount/amount_ma20' 'changepercent' 'label'， 训练时可以设置每次交易的数量，可以设置每次交易的时间间隔，可以设置每次交易的手续费，交易需要判断是否有足够的资金(USDT)，卖出时要判断是否有足够的货币量(ETH)
# 2. 用模型预测未来的数据，根据预测的结果进行交易
# 3. 计算收益率
# 4. 保存模型
# 5. 训练后的结果保存到数据库中

class TradingEnv(gym.Env):
    def __init__(self, df, keys: List[str], backtest: Backtest):
        # 初始化环境参数
        self.df = df # 历史数据
        self.keys = keys # 训练数据列
        self.backtest = backtest # 回测对象
        self.current_step = 0 # 步数计数器
        self.done = False # 是否结束标志
        self.max_step = len(self.df) - 1
        self.previous_asset = self.backtest.init_data['balance'] # 上一次的资产
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(3) # 三种动作：买入、卖出、持有
        # low = np.array([self.df[key].min() for key in keys]).min()
        # high = np.array([self.df[key].max() for key in keys]).max()
        # print(low)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(len(keys) + 3,)) 

    def reset(self):
        # 重置环境状态
        self.backtest.reset()
        self.previous_asset = self.backtest.init_data['balance']
        self.current_step = 0 
        self.done = False 
        self.action = 2
        return self._get_observation(0, 0, self.backtest.init_data['balance'])

    def step(self, action):
   
        # 执行一步动作，返回观察、奖励、是否结束和额外信息
        assert self.action_space.contains(action), "Invalid action"

        current_price = self.df.iloc[self.current_step]['close'] # 获取当前价格
        current_time = self.df.iloc[self.current_step]['time']
      
        if action == 0: # 买入
            self.backtest.mock_trade(action = "BUY", close = current_price, trade_volume = 0.1, time = current_time)
        elif action == 1: # 卖出

            self.backtest.mock_trade(action = "SELL", close = current_price, trade_volume = 0.1, time = current_time)
        self.current_step += 1
        done = self.current_step >= self.max_step

        trade_result = self.backtest.get_results()
        trade_count = trade_result.get('trade_count') or 0
        current_asset = trade_result.get('current_asset') or 0
        profit_rate = trade_result.get('profit_rate') or 0
        
        observation = self._get_observation(profit_rate, trade_count, current_asset)# 获取观察
        reward = self._get_reward(action, profit_rate, trade_count, current_asset) # 获取奖励 
        self.previous_asset = current_asset 
        # self.render(action) 

        return observation, reward, done, {}


    def _get_reward(self, action: int, profit_rate, trade_count, current_asset):
        """
        获取奖励值方法，根据当前状态和动作计算奖励值。
        """
        reward = 0
        pre_profit_rate = (current_asset - self.previous_asset) / self.previous_asset

        # 交易频率惩罚
        if action == 0 or action == 1:
            # 交易次数大于3次 惩罚
            if trade_count > 5:
                reward = -10
            elif trade_count > 3:
                reward = -5
        elif action == 2:
            if pre_profit_rate == 0:
                if profit_rate > 0:
                    reward += 10
                elif profit_rate < 0:
                    reward -= 1
            else:
                reward = pre_profit_rate * 100
        return reward

    def _get_observation(self, profit_rate, trade_count, current_asset):
        trade_obs = [profit_rate, trade_count, (current_asset - self.previous_asset) / self.previous_asset ]
        obs = np.array([self.df.iloc[self.current_step][key] for key in self.keys] + trade_obs)
        # obs = np.array([self.df.iloc[self.current_step][key] for key in self.keys])
        return obs

    def load_model(self, model_path):
        """
        加载模型方法，从指定路径加载训练好的模型。
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_profit(self):
        return self.backtest.get_results().get('profit_rate') or 0
    
    def render(self, action):
        price = self.df.iloc[self.current_step]['close'] # 获取当前价格
        # 中国时间
        time = pd.to_datetime(self.df.iloc[self.current_step]['time'])
        trade_result = self.backtest.get_results()
        profit_rate = trade_result.get('profit_rate', 0)
        trade_count = trade_result.get('trade_count', 0)
        if action == 0:
            print(f'step: {self.current_step} {price} {time} - 买入 - 收益率: {profit_rate}% {trade_count}')
        elif action == 1:
            print(f'step: {self.current_step} {price} {time} - 卖出 - 收益率: {profit_rate}% {trade_count}')
        elif action == 2:
            print(f'step: {self.current_step} {price} {time} - 持有 - 收益率: {profit_rate}% {trade_count}')
