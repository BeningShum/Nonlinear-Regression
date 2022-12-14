import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
import matplotlib.pyplot as plt  # 用于画图的库
from sklearn.model_selection import KFold  # 用于交叉验证划分数据集
from sklearn.model_selection import train_test_split  # 用于分割训练集和测试集


def RMSE(y_hat, y):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)


def RidgeRegression(
    train_x, train_y, valid_x, t
):  # train_x, train_y是训练集, valid_x是验证集, t是罚系数
    n, m = train_x.shape

    beta = (train_x.T * train_x + t * np.identity(m)).I * (
        train_x.T * train_y.reshape(n, 1)
    )

    yr = valid_x * beta
    yr = np.array(yr).reshape(
        yr.shape[0],
    )
    return yr  # yr是在valid_x上的预测值


data = pd.read_csv(".../insurance.csv")
y = data["charges"]
x = data.drop(labels="charges", axis=1)
x = pd.get_dummies(x, columns=["sex", "smoker", "region"])
cols = ["sex_female", "smoker_no", "region_northeast"]
x = x.drop(labels=cols, axis=1)
# 读入数据，将charges作为目标特征，剩余特征作为解释变量，并对分类特征进行onehot编码

x["age2"] = x["age"] ** 2
x["fat_yes"] = np.where(x["bmi"] > 30, 1, 0)
x["fat_smoker"] = x["fat_yes"] * x["smoker_yes"]
# 添加新特征，它们可能有利于解释模型
x = np.matrix(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=2022
)
# 随机选取25%的样本作为测试集，剩下的作为训练集

kf = KFold(n_splits=5, random_state=2022, shuffle=True)

T = np.linspace(0, 1, 100)
minRMSE = 1000000000
bestt = T[0]
for t in T:
    curloss = 0
    for train_x_index, valid_x_index in kf.split(
        x_train
    ):  # 进行5折交叉验证, 最终的模型评估指标为5次训练指标的平均值
        train_x = x[train_x_index]
        train_y = y[train_x_index]
        valid_x = x[valid_x_index]
        valid_y = y[valid_x_index]
        y_hat = RidgeRegression(train_x, train_y, valid_x, t)
        curloss = curloss + RMSE(y_hat, valid_y)

    if curloss / 5 < minRMSE:
        minRMSE = curloss / 5
        bestt = t

y_hat = RidgeRegression(x_train, y_train, x_test, bestt)
# 根据最佳t值进行岭回归

z = np.arange(len(y_hat))
plt.scatter(z, y_test - y_hat, facecolor="b", edgecolor="b", alpha=0.5)
plt.xticks([])
plt.title("Distribution of Residual Error")
plt.show()
# 画出残差分布图

print("RMSE of RidgeRegression is : ", RMSE(y_hat, y_test))
# 输出回归结果的RMSE
