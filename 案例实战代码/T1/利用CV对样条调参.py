import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
from sklearn.model_selection import KFold  # 用于交叉验证划分数据集
import matplotlib.pyplot as plt  # 用于画图的库

data = pd.read_csv("……/mcycle.csv")
x = data["times"]
y = data["accel"]
kf = KFold(n_splits=5, random_state=2022, shuffle=True)
# 读入数据


def RMSE(y_hat, y):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)


def CubicSplineRegression(
    train_x, train_y, valid_x, y_hat, knots
):  # 三阶样条回归, train_x, train_y是训练集；valid_x是验证集, y_hat记录测试集回归值, knots是节点
    n = len(train_x)
    K = len(knots)
    train_x = np.matrix(train_x).T
    train_y = np.matrix(train_y).T
    G = np.matrix(np.ones(n)).T  # G是设计矩阵
    for i in range(3):
        G = np.concatenate((G, np.power(train_x, i + 1)), axis=1)

    for i in knots:
        tmp = np.matrix(train_x)
        for j in range(n):
            if tmp[j] < i:
                tmp[j] = 0
            else:
                tmp[j] = tmp[j] - i
        G = np.concatenate((G, np.power(tmp, 3)), axis=1)
    # 构造设计矩阵G

    beta = (G.T * G).I * G.T * train_y  # 采用最小二乘计算参数beta
    # 估计beta

    n = len(valid_x)
    valid_x = np.matrix(valid_x).T
    for i in range(n):
        s = (
            beta[0]
            + beta[1] * valid_x[i]
            + beta[2] * valid_x[i] ** 2
            + beta[3] * valid_x[i] ** 3
        )
        for j in range(K):
            tmp = 0
            if valid_x[i] > knots[j]:
                tmp = valid_x[i] - knots[j]
            s = s + beta[j + 4] * tmp**3
        s = float(s)
        y_hat.append(s)
    # 计算验证集的估计值y_hat

    return y_hat  # 返回回归值y_hat


Knots = np.linspace(14, 16, 100)  # 调节第一个节点的位置, 大致应该在10到 16之间
minRMSE = 10000000
bestknot = [Knots[0]]  # bestknot用于记录最优节点
for knot in Knots:
    curloss = 0  # curloss记录当前的5轮测试过后的平均RMSE
    for train_x_index, valid_x_index in kf.split(x):  # 进行5折交叉验证, 最终的模型评估指标为5次训练指标的平均值
        train_x = x[train_x_index]
        train_y = y[train_x_index]
        valid_x = x[valid_x_index]
        valid_y = y[valid_x_index]
        y_hat = []
        y_hat = CubicSplineRegression(
            train_x, train_y, valid_x, y_hat, [knot, 20, 31, 36]
        )
        curloss = curloss + RMSE(y_hat, valid_y)

    if curloss / 5 < minRMSE:
        minRMSE = curloss / 5
        bestknot[0] = knot

Knots = np.linspace(19, 23, 100)
minRMSE = 10000000
bestknot.append(Knots[0])
for knot in Knots:
    curloss = 0  # curloss记录当前的5轮测试过后的平均RMSE
    for train_x_index, valid_x_index in kf.split(x):  # 进行5折交叉验证, 最终的模型评估指标为5次训练指标的平均值
        train_x = x[train_x_index]
        train_y = y[train_x_index]
        valid_x = x[valid_x_index]
        valid_y = y[valid_x_index]
        y_hat = []
        y_hat = CubicSplineRegression(
            train_x, train_y, valid_x, y_hat, [bestknot[0], knot, 31, 36]
        )
        curloss = curloss + RMSE(y_hat, valid_y)

    if curloss / 5 < minRMSE:
        minRMSE = curloss / 5
        bestknot[1] = knot

Knots = np.linspace(32, 34, 100)
minRMSE = 10000000
bestknot.append(Knots[0])
for knot in Knots:
    curloss = 0  # curloss记录当前的5轮测试过后的平均RMSE
    for train_x_index, valid_x_index in kf.split(x):  # 进行5折交叉验证, 最终的模型评估指标为5次训练指标的平均值
        train_x = x[train_x_index]
        train_y = y[train_x_index]
        valid_x = x[valid_x_index]
        valid_y = y[valid_x_index]
        y_hat = []
        y_hat = CubicSplineRegression(
            train_x, train_y, valid_x, y_hat, [bestknot[0], bestknot[1], knot, 36]
        )
        curloss = curloss + RMSE(y_hat, valid_y)

    if curloss / 5 < minRMSE:
        minRMSE = curloss / 5
        bestknot[2] = knot

Knots = np.linspace(34, 40, 100)
minRMSE = 10000000
bestknot.append(Knots[0])
for knot in Knots:
    curloss = 0  # curloss记录当前的5轮测试过后的平均RMSE
    for train_x_index, valid_x_index in kf.split(x):  # 进行5折交叉验证, 最终的模型评估指标为5次训练指标的平均值
        train_x = x[train_x_index]
        train_y = y[train_x_index]
        valid_x = x[valid_x_index]
        valid_y = y[valid_x_index]
        y_hat = []
        y_hat = CubicSplineRegression(
            train_x,
            train_y,
            valid_x,
            y_hat,
            [bestknot[0], bestknot[1], bestknot[2], knot],
        )
        curloss = curloss + RMSE(y_hat, valid_y)

    if curloss / 5 < minRMSE:
        minRMSE = curloss / 5
        bestknot[3] = knot
# 调参的方法类似坐标下降, 即先固定其他三个节点位置, 在第一个节点位置可能的范围内进行搜索, 利用交叉验证比较回归结果得到最优的第一个节点的位置；再按此法依次获得其他最优节点位置

y_hat = []
y_hat = CubicSplineRegression(x, y, x, y_hat, bestknot)
plt.scatter(x, y, facecolor="None", edgecolor="k", alpha=0.3)
plt.plot(x, y_hat, "y")
plt.legend(labels=["original data", "cubic spline optimized"])
plt.title("Optimization of the Cubic Spline Regression")
plt.show()
# 把算得的四个最优节点作为最终的节点进行拟合, 得到y_hat并画图

print("RMSE after optimization is: ", RMSE(y_hat, y))
# 调参后最终的RMSE

print("调参后, 最优的控制节点位置如下：")
for i in range(len(bestknot)):
    print("第%d个控制节点位置为: %f" % (i + 1, bestknot[i]))
# 输出最优化后的参数
