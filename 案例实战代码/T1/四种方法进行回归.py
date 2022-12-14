import pandas as pd  # 读入数据用的库
import numpy as np  # 一些科学计算工具
import matplotlib.pyplot as plt  # 用于画图的库


def RMSE(y, y_hat):  # 计算RMSE
    n = len(y)
    return np.sqrt(np.sum((y_hat - y) ** 2) / n)


def LinearSpineRegression(x, y):  # 简单线性样条回归
    n = len(x)
    x = np.matrix(x).T  # x是解释变量
    y = np.matrix(y).T  # y是因变量
    G = np.matrix(np.ones(n)).T  # G是设计矩阵
    G = np.concatenate((G, x), axis=1)
    knots = np.array([12, 23, 32, 38])  # knots是样条的节点
    for i in knots:
        tmp = np.matrix(x)
        for j in range(n):
            if tmp[j] < i:
                tmp[j] = 0
            else:
                tmp[j] = tmp[j] - i
        G = np.concatenate((G, tmp), axis=1)
    # 构造设计矩阵G

    beta = (G.T * G).I * G.T * y  # 采用最小二乘计算参数beta

    yr = []
    for i in range(n):
        s = beta[0] + beta[1] * x[i]
        for j in range(4):
            tmp = 0
            if x[i] > knots[j]:
                tmp = x[i] - knots[j]
            s = s + beta[j + 2] * tmp
        s = float(s)
        yr.append(s)
    return yr  # 返回回归值yr


def CubicSplineRegression(
    train_x, train_y, valid_x, y_hat, knots
):  # 三阶样条回归, train_x，train_y是训练集；valid_x是验证集，y_hat记录测试集回归值，knots是节点
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
    # 估计参数beta

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


def LocalPolynomialRegression(
    x, y, h, d
):  # 局部多项式回归，x是解释变量，y是响应变量，h是窗宽，d是多项式的次数（为1时表示局部线性回归）
    n = len(x)
    X = np.matrix(np.ones(n)).T
    Dmatrix = np.matrix(np.ones(n)).T
    y = np.array(y).reshape(len(y), 1)
    yr = []

    for i in range(n):
        for j in range(1, d + 1):
            xpow = np.matrix(np.power((x - x[i]), j)).T
            Dmatrix = np.concatenate((X, xpow), axis=1)  # 构造设计矩阵Dmatrix

        local_index = []  # local_index用来记录与点x[i]邻近的点的索引
        for j in range(n):
            if np.abs(x[j] - x[i]) <= h:
                local_index.append(j)

        Dmatrix_ = Dmatrix[local_index]
        y_ = y[local_index]

        Weight = np.exp(-((x - x[i]) ** 2) / (2 * h**2)) / (
            np.sqrt(2 * np.pi) * h
        )  # 这里权函数采用高斯核函数
        Weight = Weight[local_index]
        W = np.diag(Weight)

        beta = ((Dmatrix_.T * W * Dmatrix_).I) * (
            Dmatrix_.T * W * y_
        )  # 同样可以采用最小二乘得到参数beta
        yr.append(float(beta[0]))  # 回归值用yr记录

    return yr


data = pd.read_csv("……/mcycle.csv")
x = data["times"]
y = data["accel"]
# 导入数据，times是解释变量，accel是响应变量

yr1 = LinearSpineRegression(x, y)
# 简单线性样条回归的结果yr1

yr2 = CubicSplineRegression(x, y, x, [], [15, 20, 32, 40])
# 三阶样条回归的结果yr2

yr3 = LocalPolynomialRegression(x, y, 5, 1)
# 局部线性回归的结果yr3

yr4 = LocalPolynomialRegression(x, y, 10, 2)
# 局部多项式回归（次数为2）的结果yr4

plt.figure()
plt.scatter(x, y, facecolor="None", edgecolor="k", alpha=0.3)
plt.plot(x, yr1, "b")
plt.plot(x, yr2, "y")
plt.legend(
    labels=["original data", "simple spline regression", "cubic spline regression"],
    loc="upper left",
    prop={"size": 10},
)
plt.title("Spline Regression of D ataset mcycle")

plt.figure()
plt.scatter(x, y, facecolor="None", edgecolor="k", alpha=0.3)
plt.plot(x, yr3, "c")
plt.plot(x, yr4, "m")
plt.legend(
    labels=[
        "original data",
        "local linear regression",
        "local polynomial regression(quadratic)",
    ],
    loc="upper left",
    prop={"size": 8},
)
plt.title("Local Regression of Dataset mcycle")
plt.show()
# 画图

print("RMSE of simple spline regression is : ", RMSE(y, yr1))
print("RMSE of cubic spline regression is : ", RMSE(y, yr2))
print("RMSE of local linear regression is : ", RMSE(y, yr3))
print("RMSE of local polynomial regression(quadratic) is : ", RMSE(y, yr4))
# 计算四种方法的RMSE
