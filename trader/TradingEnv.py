
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
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(3) # 三种动作：买入、卖出、持有
        low = np.array([self.df[key].min() for key in keys]).min()
        high = np.array([self.df[key].max() for key in keys]).max()
        print(low, high)
        self.observation_space = spaces.Box(low=low, high=high, shape=(len(keys) + 5,)) 

    def set_df(self, df):
        self.df = df
        self.max_step = len(self.df) - 1
    def reset(self):
        # 重置环境状态
        self.backtest.reset()
        self.current_step = 0 
        self.done = False 
        self.action = 2
        return self._get_observation(0, 0, 0, 0, 0, 0)

    def step(self, action):
   
        # 执行一步动作，返回观察、奖励、是否结束和额外信息
        assert self.action_space.contains(action), "Invalid action"
        # 获取当前价格
        current_price = self.df.iloc[self.current_step]['close']
        current_open = self.df.iloc[self.current_step]['open']
        current_high = self.df.iloc[self.current_step]['high']
        current_time = self.df.iloc[self.current_step]['time']
        row = self.df.iloc[self.current_step]
        trade_volume = 600 / current_price
        trade_status = 0
        if action == 0: # 买入
            trade_status = self.backtest.mock_trade(action = "BUY", close = current_price, open = current_open, high = current_high, trade_volume = trade_volume, time = current_time)
        elif action == 1: # 卖出
            trade_status = self.backtest.mock_trade(action = "SELL", close = current_price, open = current_open, high = current_high, trade_volume = trade_volume, time = current_time)
        elif action == 2: # 持有
            self.backtest.mock_trade(action = "", close = current_price, open = current_open, high = current_high, trade_volume = 0, time = current_time)
 
        trade_result = self.backtest.get_results()
        trade_count = trade_result.get('trade_count') or 0
        current_asset = trade_result.get('current_asset') or 0
        profit_rate = trade_result.get('profit_rate') or 0
        previous_asset = trade_result.get('previous_asset', 0.001)
        position_ratio = trade_result.get('position_ratio', 0) 
        pre_profit = (current_asset - previous_asset)
        pre_profit_rate = pre_profit / previous_asset

        observation = self._get_observation(profit_rate, trade_count, pre_profit_rate, pre_profit, position_ratio, trade_status)# 获取观察
        # 亏空提前结束
        if profit_rate  < -0.98:
            return observation, profit_rate * 100, True, {}

        reward = self._get_reward(action, profit_rate, trade_count, pre_profit_rate, pre_profit, position_ratio) # 获取奖励 
        # self.render(action) 
        self.current_step += 1
        done = self.current_step >= self.max_step
        return observation, reward, done, {}


    def _get_reward(self, action: int, profit_rate, trade_count, pre_profit_rate, pre_profit, position_ratio):
        """
        获取奖励值方法，根据当前状态和动作计算奖励值。
        """
        close_ma60_rate_4h = self.df.iloc[self.current_step]['close_ma60_rate_4h']
        ma10_5m_less_than_ma30_5m = self.df.iloc[self.current_step]['ma10_5m_less_than_ma30_5m']
        ma30_5m_less_than_ma60_5m = self.df.iloc[self.current_step]['ma30_5m_less_than_ma60_5m']
        freq_penalty = 0
        hold_reward = 0
        sell_reward = 0
        # print(round(current_asset, 2), round(previous_asset, 2), round(profit_rate * 100, 2), round(pre_profit_rate * 100, 2), trade_count, action)
        # 买入惩罚
        if action == 0:
            # 交易次数大于3次 惩罚
            if trade_count > 3:
                freq_penalty = -10
            else:
                freq_penalty = 5
        # 卖出奖励、惩罚
        elif action == 1:
            # 卖出时，如果盈利，奖励
            sell_reward = pre_profit
        # 持有奖励
        elif action == 2:
            if position_ratio > 0.01:
                hold_reward = profit_rate * 100 + pre_profit
                if trade_count < 4:
                    freq_penalty += 5
            elif position_ratio < 0.01:
                hold_reward += 0.1

        return freq_penalty + hold_reward + sell_reward
    def close(self):
        # 在结束时需要将当前状态下的收益作为最终的回报传递给模型
        trade_result = self.backtest.get_results()
        profit_rate = trade_result.get('profit_rate', 0)
        return profit_rate * 100
    def _get_observation(self, profit_rate, trade_count, pre_profit_rate, pre_profit, position_ratio, trade_status):
        trade_obs = [profit_rate, pre_profit_rate, position_ratio, trade_count / 10, trade_status]
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
            print(f'step: {self.current_step} {price} {time} - 买入 - 收益率: {profit_rate * 100}% {trade_count}')
        elif action == 1:
            print(f'step: {self.current_step} {price} {time} - 卖出 - 收益率: {profit_rate * 100}% {trade_count}')
        elif action == 2:
            print(f'step: {self.current_step} {price} {time} - 持有 - 收益率: {profit_rate * 100}% {trade_count}')
