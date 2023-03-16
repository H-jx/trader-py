
import numpy as np
import gym
import pandas as pd
# 用python实现一个TradingEnv(gym.Env)，功能为通过虚拟货币(ETHUSDT)历史数据训练模型
# 1. 训练数据列： 'close' 'ma60' 'close/ma60' 'buy/amount' 'sell/amount' 'amount/amount_ma20' 'changepercent' 'label'， 训练时可以设置每次交易的数量，可以设置每次交易的时间间隔，可以设置每次交易的手续费，交易需要判断是否有足够的资金(USDT)，卖出时要判断是否有足够的货币量(ETH)
# 2. 用模型预测未来的数据，根据预测的结果进行交易
# 3. 计算收益率
# 4. 保存模型
# 5. 训练后的结果保存到数据库中

class TradingEnv3(gym.Env):
    def __init__(self, df, trade_size = 0.2, fee_rate = 0.001):
        # 初始化环境参数
        self.df = df # 历史数据
        self.trade_size = trade_size # 每次交易的数量
        self.fee_rate = fee_rate # 每次交易的手续费率
        self.balance = 1000 # 初始资金（USDT）
        self.position = 0 # 初始持仓（ETH）
        self.current_step = 0 # 步数计数器
        self.done = False # 是否结束标志
        self.max_step = len(self.df) - 1
        self.initial_asset = self.balance + self.position * self.df.iloc[0]['close']
        self.previous_asset_value = self.initial_asset
        # 定义动作空间和观察空间
        self.action_space = gym.spaces.Discrete(3) # 三种动作：买入、卖出、持有
        self.observation_space = gym.spaces.Box(low=-10, high=self.df['close'].max(), shape=(8,)) 

    def reset(self):
        # 重置环境状态
        self.balance = 1000 
        self.position = 0 
        self.current_step = 0 
        self.done = False 
        self.previous_asset_value = self.initial_asset
        return self._get_observation()

    def step(self, action):
        # 执行一步动作，返回观察、奖励、是否结束和额外信息
        assert self.action_space.contains(action), "Invalid action"
        
        price = self.df.iloc[self.current_step]['close'] # 获取当前价格

        if action == 0: # 买入
            if self.balance >= price * self.trade_size: # 判断是否有足够的资金（USDT）
                cost = price * self.trade_size * (1 + self.fee_rate) # 计算成本（USDT）
                self.balance -= cost # 更新资金（USDT）
                self.position += self.trade_size # 更新持仓（ETH）
        elif action == 1: # 卖出
            if self.position >= self.trade_size: # 判断是否有足够的持仓（ETH）
                self.balance += price * self.trade_size * (1 - self.fee_rate) # 计算收入（USDT）
                self.position -= self.trade_size# 更新持仓（ETH）

        self.current_step += 1
        done = self.current_step >= self.max_step
        observation = self._get_observation()# 获取观察
        reward = self._get_reward()# 获取奖励

        self.render(action, price)
        return observation, reward, done, {}

    def get_profit(self):
        current_asset = self.balance + self.position * self.df.iloc[self.current_step]['close']
        return (current_asset - self.initial_asset) / self.initial_asset
    def _get_reward(self, action):
        """
        获取奖励值方法，根据当前状态和动作计算奖励值。
        """
        # 当前持有的资产价值
        current_asset = self.balance + self.position * self.df.iloc[self.current_step]['close']

        # 每个时间步的奖励值为当前资产价值与上一时间步的资产价值的差值
        reward = current_asset - self.previous_asset_value

        # 更新上一时间步的资产价值
        self.previous_asset_value = current_asset

        return reward

    def _get_observation(self):
        obs = np.array([
            self.df.iloc[self.current_step]['close'],
            self.df.iloc[self.current_step]['ma60'],
            self.df.iloc[self.current_step]['close/ma60'],
            self.df.iloc[self.current_step]['buy/amount'],
            self.df.iloc[self.current_step]['sell/amount'],
            self.df.iloc[self.current_step]['amount/amount_ma20'],
            self.df.iloc[self.current_step]['changepercent'],
            self.df.iloc[self.current_step]['label']
        ])
        return obs

    def load_model(self, model_path):
        """
        加载模型方法，从指定路径加载训练好的模型。
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def load_model(self, model_path):
        """
        加载模型方法，从指定路径加载训练好的模型。
        """
        self.model.save(model_path)

    def render(self, action, close):
        price = close # 获取当前价格
        if action == 1:
            print(f'step: {self.current_step} {price} - 买入 - 收益率: {self.get_profit()}% {self._get_reward()}')
        elif action == -1:
            print(f'step: {self.current_step} {price} - 卖出 - 收益率: {self.get_profit()}% {self._get_reward()}')