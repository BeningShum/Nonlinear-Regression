import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
import matplotlib.pyplot as plt  # 用于画图的库
from sklearn.model_selection import train_test_split  # 用于分割测试集和训练集


def RMSE(y_hat, y):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)


def KernelSmoothing(
    x_train, y_train, x_test, h
):  # 核平滑，x_train、y_train是训练集，x_test是测试集，h是窗宽
    n, m = x_train.shape  # n是训练集样本容量，m是特征数量
    N = x_test.shape[0]  # N是测试集样本容量
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)

    yr = []  # yr记录测试集的预测结果
    for i in range(N):
        local_index = []  # local_index寻找与当前测试样本邻近的训练样本
        for j in range(n):
            if (
                np.linalg.norm((x_train[j] - x_test[i]).reshape(m, 1), ord=1, axis=0)
                <= h
            ):
                local_index.append(j)

        Weight = np.exp(
            -(np.linalg.norm(x_train - x_test[i], ord=1, axis=1) ** 2) / (2 * h**2)
        ) / (
            np.sqrt(2 * np.pi) * h
        )  # 权函数采用高斯核函数
        Weight = Weight[local_index]

        y_hat = float(
            np.dot(Weight.reshape(1, len(Weight)), y_train[local_index])
            / np.sum(Weight)
        )
        yr.append(y_hat)

    return yr


data = pd.read_csv("C:\\Users\\16548\\Desktop\\导论作业\\T2\\insurance.csv")
y = data["charges"]
x = data.drop(labels="charges", axis=1)
x = pd.get_dummies(x, columns=["sex", "smoker", "region"])
cols = ["sex_female", "smoker_no", "region_northeast"]
x = x.drop(labels=cols, axis=1)
# 读入数据，将charges作为目标特征，剩余特征作为解释变量，并对分类特征进行onehot编码

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=2022
)
# 将完整数据集按1：3的比例分割为测试集和训练集

y_hat_1 = KernelSmoothing(x_train, y_train, x_test, 4)
y_hat_2 = KernelSmoothing(x_train, y_train, x_test, 8)
y_hat_3 = KernelSmoothing(x_train, y_train, x_test, 2)
# 核平滑，为了对比窗宽的影响，选取三个窗宽进行平滑

T = np.arange(len(y_test))
fig = plt.figure()
figsub = fig.add_subplot(211)
figsub.scatter(T, y_test - y_hat_1, facecolor="m", edgecolor="m", alpha=0.5)
figsub.set_title("h=4, RMSE=%.2f" % RMSE(y_hat_1, y_test))

figsub = fig.add_subplot(212)
figsub.scatter(T, y_test - y_hat_2, facecolor="b", edgecolor="b", alpha=0.5)
figsub.set_title("h=8, RMSE=%.2f" % RMSE(y_hat_2, y_test))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.xticks([])

fig = plt.figure()
figsub = fig.add_subplot(211)
figsub.scatter(T, y_test - y_hat_1, facecolor="m", edgecolor="m", alpha=0.5)
figsub.set_title("h=4, RMSE=%.2f" % RMSE(y_hat_1, y_test))


figsub = fig.add_subplot(212)
figsub.scatter(T, y_test - y_hat_3, facecolor="r", edgecolor="r", alpha=0.5)
figsub.set_title("h=2, RMSE=%.2f" % RMSE(y_hat_3, y_test))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.xticks([])
plt.show()
# 画图进行对比
