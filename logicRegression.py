import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# 梯度上升法
def logicRegression(x, y, alpha, iter):
    numSamples, numFeatures = np.shape(x)
    weights = np.ones((numFeatures, 1))
    for i in range(iter):
        fx = x * weights
        hx = sigmoid(fx)
        weights = weights + alpha * x.T * (y - hx)
    return weights
# 随机梯度上升法
def stochLogicRegression(x, y, alpha, iter):
    numSamples, numFeatures = np.shape(x)
    weights = np.ones((numFeatures, 1))
    for i in range(iter):
        for j in range(numSamples):
            fx = x[j, :] * weights
            hx = sigmoid(fx)
            weights = weights + alpha * x[j, :].T * (y[j, :] - hx)
    return weights
# 在迭代好的回归因子下计算模型在训练数据上表现的准确度
def accLogicRegression(weights, x, y):
    numSamples, numFeatures = np.shape(x)
    accuracy = 0.0
    for i in range(numSamples):
        predict = sigmoid(x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(y[i, 0]):
            accuracy += 1
    print('逻辑回归模型准确率为{0}%'.format(accuracy / numSamples * 100))
# 可视化二维训练数据的分类结果
def showLogicRegression(weights, x, y):
    numSamples, numFeatures = np.shape(x)
    for i in range(numSamples):
        if int(y[i, 0]) == 0:
            plt.plot(x[i, 1], x[i, 2], 'om')
        elif int(y[i, 0]) == 1:
            plt.plot(x[i, 1], x[i, 2], 'ob')
    xa1 = min(x[:, 1])[0, 0]
    xb1 = max(x[:, 1])[0, 0]
    xa2 = - ((weights[0] + weights[1] * xa1) / weights[2]).tolist()[0][0]
    xb2 = - ((weights[0] + weights[1] * xb1) / weights[2]).tolist()[0][0]
    plt.plot([xa1, xb1], [xa2, xb2], '#FB4A42')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
def loadData(src):
    x, y = [], []
    lineArr = []
    with open(src) as fileIn:
        [lineArr.append(line.strip().split(',')) for line in fileIn.readlines()]
    np.random.shuffle(lineArr)  # 随机打乱训练数据集
    for line in lineArr:
        x.append([1.0, float(line[0]), float(line[1])])
        y.append(float(line[2]))
    return np.mat(x), np.mat(y).T
if __name__ =='__main__':
    x, y = loadData('./data/data.csv')
    # weights = logicRegression(x, y, alpha=0.01, iter=500)
    weights = stochLogicRegression(x, y, alpha=0.01, iter=200)
    accLogicRegression(weights, x, y)
    # showLogicRegression(weights, x, y)
