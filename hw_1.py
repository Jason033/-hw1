import pandas as pd

# 读取数据
data = pd.read_csv('mydata.csv')

# 数据理解
print(data.info())
print(data.describe())
import numpy as np

def generate_data(a, b, num_points, noise_level):
    x = np.linspace(0, 10, num_points)
    y = a * x + b + noise_level * np.random.randn(num_points)
    return pd.DataFrame({'x': x, 'y': y})
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def fit_model(data):
    model = LinearRegression()
    X = data['x'].values.reshape(-1, 1)
    y = data['y']
    model.fit(X, y)
    return model
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, data):
    X = data['x'].values.reshape(-1, 1)
    y_true = data['y']
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2
import streamlit as st

# Web 界面
st.title("线性回归模型")

# 用户输入
a = st.slider('选择斜率 (a)', 0.0, 10.0, 1.0)
b = st.slider('选择截距 (b)', 0.0, 10.0, 1.0)
num_points = st.slider('选择数据点数量', 10, 100, 50)
noise_level = st.slider('选择噪声水平', 0.0, 5.0, 1.0)

# 生成数据
data = generate_data(a, b, num_points, noise_level)
st.write(data)

# 拟合模型并展示结果
model = fit_model(data)
mse, r2 = evaluate_model(model, data)
st.write(f"均方误差: {mse}")
st.write(f"R²: {r2}")

# 绘图
plt.scatter(data['x'], data['y'], label='Data Points')
plt.plot(data['x'], model.predict(data['x'].values.reshape(-1, 1)), color='red', label='Fitted Line')
plt.legend()
st.pyplot(plt)

