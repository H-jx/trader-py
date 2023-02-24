# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('data/btcusdt-4hour-2021-01-18.xlsx')

# 删除不必要的列
df = df.drop(columns=['vol', 'count', 'low-close/close', 'high-close/close'])
print(df.head(5))
# 使用 one-hot 编码将标签转换为数字
df['label'] = pd.Categorical(df['label'])
df['label'] = df['label'].cat.codes

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

# 对数据进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义神经网络模型
def neural_network_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 训练神经网络
input_shape = (X_train.shape[1],)
model = neural_network_model(input_shape)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 进行交易预测
test_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
test_data = scaler.transform(test_data)
prediction = model.predict(test_data)
print('Prediction for test data:', prediction)
