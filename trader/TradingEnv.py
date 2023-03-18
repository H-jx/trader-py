
import pickle
from typing import List

import gym
import numpy as np
import pandas as pd

# 用python实现一个TradingEnv(gym.Env)，功能为通过虚拟货币(ETHUSDT)历史数据训练模型
# 1. 训练数据列： 'close' 'ma60' 'close/ma60' 'buy/amount' 'sell/amount' 'amount/amount_ma20' 'changepercent' 'label'， 训练时可以设置每次交易的数量，可以设置每次交易的时间间隔，可以设置每次交易的手续费，交易需要判断是否有足够的资金(USDT)，卖出时要判断是否有足够的货币量(ETH)
# 2. 用模型预测未来的数据，根据预测的结果进行交易
# 3. 计算收益率
# 4. 保存模型
# 5. 训练后的结果保存到数据库中

class TradingEnv(gym.Env):
    def __init__(self, df, keys: List[str], trade_size = 0.2, fee_rate = 0.001):
        # 初始化环境参数
        self.df = df # 历史数据
        self.keys = keys # 训练数据列
        self.trade_size = trade_size # 每次交易的数量
        self.fee_rate = fee_rate # 每次交易的手续费率
        self.balance = 1000 # 初始资金（USDT）
        self.position = 0 # 初始持仓（ETH）
        self.initial_asset = self.balance + self.position * self.df.iloc[0]['close']
        self.previous_asset_value = self.initial_asset
        self.current_step = 0 # 步数计数器
        self.done = False # 是否结束标志
        self.max_step = len(self.df) - 1

        # 定义动作空间和观察空间
        self.action_space = gym.spaces.Discrete(3) # 三种动作：买入、卖出、持有
        low = np.array([self.df[key].min() for key in keys]).min()
        high = np.array([self.df[key].max() for key in keys]).max()

        self.observation_space = gym.spaces.Box(low=int(low), high=int(high), shape=(len(keys) + 4,)) 


    def reset(self):
        # 重置环境状态
        self.balance = 1000 
        self.position = 0 
        self.current_step = 0 
        self.done = False 
        self.previous_asset_value = self.initial_asset
        self.action = 2
        self.last_trade_time = self.df.iloc[self.current_step]['timestamp']
        return self._get_observation(self.action)

    def step(self, action):
   
        # 执行一步动作，返回观察、奖励、是否结束和额外信息
        assert self.action_space.contains(action), "Invalid action"

        price = self.df.iloc[self.current_step]['close'] # 获取当前价格
        can_trade = self.df.iloc[self.current_step]['timestamp'] - self.last_trade_time > 900
        # can_trade = True
        can_buy = self.balance >= price * self.trade_size
        can_sell = self.position >= self.trade_size
        action_text = 'hold'
        if action == 0 and can_trade: # 买入
            if can_buy: # 判断是否有足够的资金（USDT）
                cost = price * self.trade_size * (1 + self.fee_rate) # 计算成本（USDT）
                self.balance -= cost # 更新资金（USDT）
                self.position += self.trade_size # 更新持仓（ETH）
                self.last_trade_time = self.df.iloc[self.current_step]['timestamp']
                action_text = 'buy'
        elif action == 1 and can_trade: # 卖出
            if can_sell: # 判断是否有足够的持仓（ETH）
                self.balance += price * self.trade_size * (1 - self.fee_rate) # 计算收入（USDT）
                self.position -= self.trade_size# 更新持仓（ETH）
                self.last_trade_time = self.df.iloc[self.current_step]['timestamp']
                action_text = 'sell'
        self.current_step += 1
        done = self.current_step >= self.max_step
        observation = self._get_observation(action)# 获取观察
        reward = self._get_reward(action)# 获取奖励

        # 如果action == 0 can_buy == False，reward = 0, 如果action == 1 can_sell == False，reward = 0
        if action == 0 and not can_buy:
            reward = 0
            action = 2
        elif action == 1 and not can_sell:
            reward = 0
            action = 2
        else:
            reward = self._get_reward(action)# 获取奖励
        # print(self.df.iloc[self.current_step]['time'], action, reward)
        # self.render(action, can_trade)
        return observation, reward, done, {"action": action_text, "time": self.df.iloc[self.current_step]['time'], "price": price}

    def get_profit(self):
        current_asset = self.balance + self.position * self.df.iloc[self.current_step]['close']
        return (current_asset - self.initial_asset) / self.initial_asset
    def _get_reward(self, action):
        """
        获取奖励值方法，根据当前状态和动作计算奖励值。
        """
        if action == 2:
            return 0
        # 当前持有的资产价值
        current_asset = self.balance + self.position * self.df.iloc[self.current_step]['close']

        # 每个时间步的奖励值为当前资产价值与上一时间步的资产价值的差值
        reward = current_asset - self.previous_asset_value

        # 更新上一时间步的资产价值
        self.previous_asset_value = current_asset

        return reward

    def _get_observation(self, action):
        ##  添加一个self.get_profit(), 作为观察值
        obs = np.array([self.df.iloc[self.current_step][key] for key in self.keys] + [action, self.get_profit(), self.balance, self.position])
        # obs = np.array([self.df.iloc[self.current_step][key] for key in self.keys])
        return obs

    def load_model(self, model_path):
        """
        加载模型方法，从指定路径加载训练好的模型。
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def render(self, action):
        price = self.df.iloc[self.current_step]['close'] # 获取当前价格
        # 中国时间
        time = pd.to_datetime(self.df.iloc[self.current_step]['time'])
        if action == 0:
            print(f'step: {self.current_step} {price} {time} - 买入 - 收益率: {round(self.get_profit(), 4) * 100}%')
        elif action == 1:
            print(f'step: {self.current_step} {price} {time} - 卖出 - 收益率: {round(self.get_profit(), 4) * 100}%')
