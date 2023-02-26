import numpy as np
import pandas as pd
from stable_baselines3 import DQN
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model

from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, data):
        self.df = data
        self.reward_range = (-1, 1)
        self.action_space = gym.spaces.Discrete(3)  # 买入，卖出，不操作
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        
    def reset(self):
        self.current_step = 0
        self.done = False
        self.profit = 0
        self.position = 0
        self.history = deque(maxlen=60)
        self.history.extend([self.df.iloc[0]['close']] * 60)
        return self._next_observation()

    def step(self, action):
        if action == 0:  # 买入
            if self.position == 1:
                reward = -0.1
            else:
                reward = 0
                self.position = 1
                self.entry_price = self.df.iloc[self.current_step]['close']
        elif action == 1:  # 卖出
            if self.position == -1:
                reward = -0.1
            else:
                reward = self._get_reward(self.df.iloc[self.current_step]['close'])
                self.position = -1
                self.profit += reward
        else:  # 不操作
            reward = 0
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        return self._next_observation(), reward, self.done, {}

    def _next_observation(self):

        obs = np.array([
            self.df.iloc[self.current_step]['close/ma60'],
            self.df.iloc[self.current_step]['buy/amount'],
            self.df.iloc[self.current_step]['sell/amount'],
            self.df.iloc[self.current_step]['amount/amount_ma20'],
            self.df.iloc[self.current_step]['changepercent'],
            self.df.iloc[self.current_step]['label']
        ])
        return obs

    def _get_reward(self, current_price):
        reward = 0
        if self.position == 1:
            reward = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            reward = (self.entry_price - current_price) / self.entry_price
        return reward

    def render(self, mode='human', close=False):
        profit = round(self.profit * 100, 2)
        current_price = self.df.iloc[self.current_step]['close']
        if self.position == 1:
            print(f'step: {self.current_step} {current_price} - 持有中 - 收益率: {profit}%')
        elif self.position == -1:
            print(f'step: {self.current_step} {current_price} - 空仓中 - 收益率: {profit}%')
        else:
            print(f'step: {self.current_step} {self.df.iloc[self.current_step]} - 空仓中 - 收益率: {profit}%')
    def load_model(path: str):
        # 加载已保存的模型
        model = tf.keras.models.load_model(path)

        # 将模型传递给 Gym 环境
        env = gym.make('MyEnv')
        env.set_model(model)
    def save_model(self, path, model):
            save_model(model, path)
            print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path):
        model = tf.keras.models.load_model(path)
        return cls(model)