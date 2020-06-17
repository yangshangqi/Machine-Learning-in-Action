# autor: zhumenger
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #将x0设为1 存储前俩个数据x1, x2
        labelMat.append(int(lineArr[2])) # 存储标签
    return dataMat,labelMat
   
def sigmoid(inX): # 阶跃函数，输入数据的类别(0或1)
    return 1.0/(1+exp(-inX))

#利用梯度上升法找到最佳回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             # 将列表转化为numpy矩阵
    labelMat = mat(classLabels).transpose() # 将行向量转变成列向量
    m,n = shape(dataMatrix) # 得到矩阵的行数和列数
    alpha = 0.001 # 设置目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n,1)) # 先将所有的回归系设为1
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     # 矩阵相乘之后得到的是一个列向量
        error = (labelMat - h)              # 计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose()* error #按照该差值的方向调整回归系数
    return weights

# 根据回归系数确定不同类别数据之间的分割线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] # 得到行数
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') # 画点 红色， 方形
    ax.scatter(xcord2, ycord2, s=30, c='green') # 绿色 圆形
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
# 利用随机梯度上升法求出回归系数（一种在线学习算法，可进行增量式更新，用来处理大数据）
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix) # 得到数据的行数和列数
    alpha = 0.01
    weights = ones(n)   # 开始时将回归系数初始化为1
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h  # 计算差值
        weights = weights + alpha * error * dataMatrix[i] # 按照该差值的方向调整回归系数
    return weights

# 对随机梯度上升法进行改进
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter): # 迭代次数
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # 每次迭代都要调整，缓解数据波动
            randIndex = int(random.uniform(0,len(dataIndex)))# 随机选取样本，用来减少周期性的波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) # 删除该样本
    return weights


# Logistic回归分类函数

# 使用sigmoid函数得到该数据的类别标签
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 对训练集和测试集的数据进行预处理，返回该模型的错误率
def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21): # 将每个数据的前20个特征提取出来
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) # 提取该数据的标签
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) # 得到最优回归系数
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) # 得到错误率
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


# 得到错误率的平均值
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))