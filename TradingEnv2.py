import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TradingEnv2(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        self.symbols = ['ETHUSDT']
        self.df = data
        self.n_step = self.df.shape[0]
        self.current_step = None
        self.trade_volume = 0.1
        self.action_space = gym.spaces.Discrete(3)  # 买入，卖出，不操作
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))
        self.reset()
    def reset(self):
          # 重置当前步骤和初始资本，持仓
        self.current_step = 0
        self.initial_capital = 1000.0
        self.initial_holding = {'ETHUSDT': 0.0}
        return self._next_observation()

    def step(self, action):
         # 执行action并返回reward和下一个状态
        if action == 0:
            # hold
            reward = 0
        elif action == 1:
            # buy
            reward = -0.001
            price = self.df.loc[self.current_step, 'close']
            volume = self.trade_volume
            self.current_holding['ETHUSDT'] += volume
            self.current_capital -= price * volume
        elif action == 2:
            # sell
            reward = 0.001
            price = self.df.loc[self.current_step, 'close']
            volume = min(self.current_holding['ETHUSDT'], self.trade_volume)
            self.current_capital += price * volume
            self.current_holding['ETHUSDT'] -= volume
        self.current_step += 1
        done = self.current_step >= self.n_step
        info = {}
        return self._next_observation(), reward, done, info

    def _next_observation(self):

        obs = np.array([
            self.df.iloc[self.current_step]['close/ma60'],
            self.df.iloc[self.current_step]['buy/amount'],
            self.df.iloc[self.current_step]['sell/amount'],
            self.df.iloc[self.current_step]['amount/amount_ma20'],
            self.df.iloc[self.current_step]['changepercent'],
            # self.df.iloc[self.current_step]['label']
        ])
        return obs

    def _get_reward(self, current_price):
        reward = 0
        if self.position == 1:
            reward = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            reward = (self.entry_price - current_price) / self.entry_price
        return reward

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Capital: {self.current_capital}')
        print(f'Holding: {self.current_holding}')
