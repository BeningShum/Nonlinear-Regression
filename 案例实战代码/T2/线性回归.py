import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
import matplotlib.pyplot as plt  # 用于画图的库
from sklearn.model_selection import train_test_split  # 用于分割训练集和测试集


def RMSE(y_hat, y):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)


def LinearRegression(train_x, train_y, test_x):
    n = train_x.shape[0]

    beta = (train_x.T * train_x).I * (train_x.T * train_y.reshape(n, 1))

    yr = test_x * beta
    yr = np.array(yr).reshape(yr.shape[0],)
    return yr  # yr是在valid_x上的预测值


data = pd.read_csv(".../insurance.csv")
y = data["charges"]
x = data.drop(labels="charges", axis=1)
x = pd.get_dummies(x, columns=["sex", "smoker", "region"])
cols = ["age", "bmi", "smoker_yes"]
x = x[cols]
x = np.matrix(x)
y = np.array(y)
# 读入数据，将charges作为目标特征，剩余特征作为解释变量，并对分类特征进行onehot编码

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, random_state=2022
)
# 随机选取25%的样本作为测试集，剩下的作为训练集

y_hat = LinearRegression(train_x, train_y, test_x)
# 根据最佳t值进行岭回归

z = np.arange(len(y_hat))
plt.scatter(z, test_y - y_hat, facecolor="r", edgecolor="r", alpha=0.5)
plt.xticks([])
plt.title("Distribution of Residual Error")
plt.show()
# 画出残差分布图

print("RMSE of LinearRegression is : ", RMSE(y_hat, test_y))
# 输出回归结果的RMSE
