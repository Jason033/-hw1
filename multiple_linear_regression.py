import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 1: Load data
data = pd.read_csv('multiple_linear_regression.csv')

# 確認資料是否載入成功
print(data.head())

# 生成隨機資料的散佈圖 (例: R&D Spend vs Profit)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['R&D Spend'], y=data['Profit'])
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
st.pyplot(plt)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Step 2: Preprocessing

# One-hot encoding for categorical feature "State"
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# X 為所有的自變數，y 為目標變數 Profit
X = data.drop('Profit', axis=1)
y = data['Profit']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 檢查資料型態和重塑
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
from sklearn.linear_model import LinearRegression

# Step 3: Build Model

# 初始化線性迴歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集的 profit
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 4: Evaluation

# 計算評估指標
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# 評估過擬合與欠擬合
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training score: {train_score}")
print(f"Test score: {test_score}")

# 畫出 training 和 test curve
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Profit')
plt.plot(y_pred, label='Predicted Profit', linestyle='--')
plt.title('Actual vs Predicted Profit')
plt.xlabel('Data Points')
plt.ylabel('Profit')
plt.legend()
st.pyplot(plt)
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def objective(trial):
    # 定義模型參數範圍
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    
    # 標準化數據
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 建立模型
    model = LinearRegression(fit_intercept=fit_intercept)
    
    # 使用交叉驗證評估模型
    score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

# 初始化Optuna的study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 查看最佳參數
print("Best parameters: ", study.best_params)

import numpy as np

# Step 5: Deployment on web (Streamlit)
def encode_state(state):
    # 假設我們在訓練時使用的 states 為 ['California', 'New York', 'Florida']
    if state == 'California':
        return [1, 0]
    elif state == 'New York':
        return [0, 1]
    else:  # Florida
        return [0, 0]
# 預測功能
def predict_profit(r_d_spend, administration, marketing_spend, state):
    # 構建輸入資料
    state_encoded = encode_state(state)
    input_data = np.array([[r_d_spend, administration, marketing_spend] + state_encoded])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit 網頁設計
st.title("Multiple Linear Regression Web App")
st.write("預測 Profit")

# 建立輸入介面
r_d_spend = st.number_input("R&D Spend", value=0.0)
administration = st.number_input("Administration", value=0.0)
marketing_spend = st.number_input("Marketing Spend", value=0.0)
state = st.selectbox("State", options=['California', 'New York', 'Florida'])

# 預測結果
if st.button("Predict"):
    result = predict_profit(r_d_spend, administration, marketing_spend, state)
    st.success(f"預測的 Profit 為: {result}")
