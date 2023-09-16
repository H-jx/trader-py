import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 读取金融交易数据
df = pd.read_csv('data/sample-data.csv')

# 数据预处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[[
    'open', 
    'high', 
    'low', 
    'close', 
    'close_less_than_ma10_5m',
    'ma10_5m_less_than_ma30_5m',
    'ma30_5m_less_than_ma60_5m',
    'change_rate_5m',
    'close_ma60_rate_5m',
    'close_ma60_rate_4h',
    'volume_ma20_rate_5m',
    'volume_ma20_rate_4h',
    'rsi_4h',
    'close_ma30_rate_1d'
]])
x_train = []
y_train = []
for i in range(60, len(df.tail(1000))):
    x_train.append(scaled_data[i-60:i])
    y_train.append(scaled_data[i, 3])
x_train, y_train = np.array(x_train), np.array(y_train)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 14)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

# 训练模型
model.fit(x_train, y_train, epochs=100, batch_size=32)

# 预测新数据
last_60_days = scaled_data[-60:]
last_60_days_scaled = scaler.transform(last_60_days)
x_test = np.array([last_60_days_scaled])
y_pred = model.predict(x_test)

# 反向缩放预测结果
predicted_price = scaler.inverse_transform(np.array([[y_pred[0][0], 0, 0, 0, 0]]))

# 打印预测结果
print(predicted_price[0][0])
