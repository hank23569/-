import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit應用開始
st.title("簡單線性回歸模擬")

# 用戶輸入
a = st.slider("調整斜率 (a)", -10.0, 10.0, 1.0)
b = st.slider("調整截距 (b)", -10.0, 10.0, 0.0)
noise_level = st.slider("噪音大小", 0.0, 10.0, 1.0)
num_points = st.slider("數據點數量", 10, 100, 50)

# 生成數據
np.random.seed(42)  # 設定隨機種子以便重現
X = np.linspace(0, 10, num_points)
y_true = a * X + b
y = y_true + np.random.normal(0, noise_level, num_points)

# 線性回歸建模
model = LinearRegression()
X_reshaped = X.reshape(-1, 1)
model.fit(X_reshaped, y)
y_pred = model.predict(X_reshaped)

# 計算均方誤差
mse = mean_squared_error(y, y_pred)

# 繪製散點圖和回歸線
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='數據點')
plt.plot(X, y_pred, color='red', label='回歸線', linewidth=2)
plt.title('線性回歸')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
st.pyplot(plt)

# 顯示模型參數
st.write(f"模型斜率: {model.coef_[0]:.2f}")
st.write(f"模型截距: {model.intercept_:.2f}")
st.write(f"均方誤差 (MSE): {mse:.2f}")
