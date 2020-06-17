# autor: zhumenger

# 2.1 K近邻算法
from os import listdir
from numpy import *
import operator
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
group, labels = createDataSet() #调用方法
group.shape[0] #shape(i)获得 i + 1维数组中有多少个元素

#输入向量inX， 输入的训练样本集dataSet, 标签向量labels，k表示最近邻的数目

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #获得第 i + 1 维的长度
    #tile用法：表示inX这个列表， 第一个维度重复dataSetSize遍， 第二个维度重复1遍
    #diffMat表示inX这个集合到dataSet的差是多少
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2 #求出该矩阵内数各个值的的平方
    sqDistances = sqDiffMat.sum(axis=1)# sum(axis = 0)计算每一列的和sum(axis = 1)计算每一行的和
    distances = sqDistances**0.5   #开根号  12 ~ 18 就是在求sqrt((x1 - x2)**2 + (y1 - y2)**2)
    sortedDistIndicies = distances.argsort() #注意argsort()它不会改变原来的数组，它返回一个新的数组，它的数据元素是（默认是从小到大）索引值
    classCount={}          #字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #这一步就是在求哪个标签出现的次数最多
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #get函数：在字典中寻找voteIlabel的值， 如果不存在返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)# 将字典中的值从小到大排序
    return sortedClassCount[0][0]  # 获得字典中第一个元素所对应的值

#2.2 使用K近邻算法改进约会网站的配对效果

# 2.2.1 创建文件矩阵函数：将文本中的数据转化到矩阵中
def file2matrix(filename):
    fr = open(filename)
    # read( )：表示读取全部内容当无参数的时候      readline( )：表示每次读出一行   readlines( )是读取所有的行
    # 每一行作为一个元素放在列表里面 。 所以arrayOLines为列表
    arrayOLines = fr.readlines();
    numberOfLines = len(arrayOLines)         #得到文件的行数
    returnMat = zeros((numberOfLines,3))        #创建一个有 number行， 每行有 3 个元素的矩阵， 默认值为0
    classLabelVector = []                       #将数据中的信息解析到列表
    index = 0
    for line in arrayOLines:
        line = line.strip() #strip() 方法用于移除字符串头尾指定的字符（默认为空格)
        listFromLine = line.split('\t') #将字符串line以制表符分割成列表
        returnMat[index,:] = listFromLine[0:3] # 将listFromLine中的前3个元素加入到returnMat[index]中
        classLabelVector.append(int(listFromLine[-1])) #获得数据列表中的最后一个元素
        index += 1
    return returnMat,classLabelVector

# 2.2.3 归一化特征值
# 特征值的数值有大有小,通过此函数可以将数字特征值转化为0到1的区间
def autoNorm(dataSet):
    minVals = dataSet.min(0)# min(0)返回该矩阵中每一列的最小值 min(1)返回该矩阵中每一行的最小值
    maxVals = dataSet.max(0)# 得到的是一个矩阵
    ranges = maxVals - minVals # 让最大值-最小值
    normDataSet = zeros(shape(dataSet)) # 创建矩阵
    print(normDataSet)
    m = dataSet.shape[0] #获得第一维的元素个数
    normDataSet = dataSet - tile(minVals, (m,1)) #当前值减去最小值， tile()函数，将矩阵minVals复制m份
    normDataSet = normDataSet/tile(ranges, (m,1)) #将数字特征值转化为了0到1的区间
    return normDataSet, ranges, minVals

#2.2.4 分类器针对约会网站的测试代码

#拿出 10% 的数据去测试该算法的出错率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #得到测试用的数据数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

# 2.2.5 约会网站预测函数
def classifyPerson():
    resultList = ['largeDos es', 'smallDoses', 'didntLike']  # 3种预测结果集
    percentTats = float(input("Please input percentage of time spent playing vedio games?"))
    ffMiles = float(input("frequent flier miles earned consumed per year?"))
    iceCream = float(input("Liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person : %s " % resultList[classifierResult - 1])

#2.3 手写识别系统

#2.3.1 准备数据：将图像转换为单行测试向量
def img2vector(filename):
    returnVect = zeros((1,1024)) #构建矩阵
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 2.3.2使用k近邻算法识别手写数字
# 测试代码，查看错误率
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')     #得到文本列表
    m = len(trainingFileList)   #获取长度
    trainingMat = zeros((m,1024)) # 初始化矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]# 得到第i个文本
        fileStr = fileNameStr.split('.')[0]     #以小数点分割字符串，并得到第 0 个元素
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #将图像转换为单行测试向量
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')      #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #使用K近邻算法
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))