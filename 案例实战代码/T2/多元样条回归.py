from pyearth import Earth  # 多元样条回归库
import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
import matplotlib.pyplot as plt  # 用于画图的库
from sklearn.model_selection import train_test_split  # 用于分割训练集和测试集


def RMSE(y_hat, y):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)  # 计算R^2


def Rsquare(y_hat, y):
    n = len(y)
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


data = pd.read_csv("C:\\Users\\16548\\Desktop\\导论作业\\T2\\insurance.csv")
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

MARS = Earth(smooth=False)
# 构建多元自适应样条模型，设定smooth参数为True，使得一阶偏导数连续

MARS.fit(x_train, y_train)
# 训练模型

y_hat = MARS.predict(x_test)
# y_hat用于记录估计结果

z = np.arange(len(y_hat))
plt.scatter(z, y_test - y_hat, facecolor="r", edgecolor="r", alpha=0.5)
plt.xticks([])
plt.title("Distribution of Residual Error")
plt.show()
# 画出残差分布图

print("RMSE of MARS model is : ", RMSE(y_hat, y_test))
print("R square of MARS model is : ", Rsquare(y_hat, y_test))
# 输出模型评估指标: RMSE和判定系数R^2
