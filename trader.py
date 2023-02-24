import backtrader as bt
import tensorflow as tf
import numpy as np

class NeuralNetworkStrategy(bt.Strategy):
    params = (
        ('lookback', 10),
        ('n_hidden', 10),
        ('lr', 0.001),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.params.n_hidden, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.params.lr), loss='mean_squared_error')
        return model

    def next(self):
        if len(self.data_close) < self.params.lookback:
            return

        # Input data
        x = np.array([self.data_close[i] for i in range(-self.params.lookback, 0)]).reshape(1, -1)

        # Predict next price
        y_pred = self.model.predict(x)[0][0]

        # Buy or sell based on predicted price
        if self.data_close[0] < y_pred:
            self.buy(size=1)
        elif self.data_close[0] > y_pred:
            self.sell(size=1)

        # Train the model on the current close price
        x_train = np.array([self.data_close[i] for i in range(-self.params.lookback, 0)]).reshape(-1, self.params.lookback)
        y_train = np.array([self.data_close[0]])
        self.model.train_on_batch(x_train, y_train)

daming = NeuralNetworkStrategy(bt)