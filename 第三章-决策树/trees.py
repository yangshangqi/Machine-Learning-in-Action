# autor: zhumenger

'''
决策树：
根据原始数据构造决策树，可以将不熟悉的数据集合提取出一系列规则,
该代码块包含几个部分：
    1.得到最好的划分方式：
        给你一堆初始信息，首先判断应该从哪个特征划分数据分类
        划分数据的原则应为：将无序的数据变得更加有序，这里使用了信息的香农熵，
        利用该公式求出的当前信息的香农熵与初始的香农熵之差，值越小，无序度就会越低，
        以此划分的方式就会更好
    2.以最好的划分方式得到的数据集构建决策树：
        通过递归的方式构建决策树
'''

from math import log
import operator
#3-1： 计算给定的数据集的香农熵，用来度量数据集的无序程度，即计算信息的期望值∑(1 - n)p[i] * log2(p[i])
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #获取数据的总数
    labelCounts = {}    #字典用来存最后一个数出现的次数
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries #求频率
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt #返回期望值, 期望值越高，则混合的数据也越多

#数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


#3-2：按照给定特征划分数据集

#第一个参数为数据集，第二个参数为划分数据集的特征，第三个参数为要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


#3-3：选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet) #获得信息的期望值，熵
    bestInfoGain = 0.0;
    bestFeature = -1
    #按照第i个特征值对其进行划分的情况
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  #将数据集中每个元素的第i个元素特征值存储到一个列表中
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)#对每个属性的每个特征值进行一次划分
            prob = len(subDataSet) / float(len(dataSet)) #获得该划分方式的占比情况
            newEntropy += prob * calcShannonEnt(subDataSet) #对所有唯一特征值得到的熵求和
        infoGain = baseEntropy - newEntropy  #信息增益：熵的减少即数据无序度的减少，这个值越大，数据越有序，说明该划分方式更好
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i  #记录该特征值
    return bestFeature  # returns an integer

#返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#3-4：创建树的函数代码
#俩个参数，第一个为数据集，第二个是标签列表
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  #得到数据集中每一个元素列表的最后一个特征元素
    if classList.count(classList[0]) == len(classList):
        return classList[0]  #如果特征值都相同，则返回
    #如果只有一个特征值，则返回
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList) #返回出现次数最多的特征值
    bestFeat = chooseBestFeatureToSplit(dataSet) #找到最好的划分方式
    bestFeatLabel = labels[bestFeat] #记录该标签
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet] #获得这一列的所有特征值
    uniqueVals = set(featValues) #去重
    for value in uniqueVals:  #对于每一个特征值划分数据
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#3.3.1测试算法：使用决策树执行分类
#3-8.使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]   #获得第一个标签的名字
    secondDict = inputTree[firstStr]     #获得以一个节点的分支
    featIndex = featLabels.index(firstStr) #获得该标签的下标
    # key = testVec[featIndex]
    # valueOfFeat = secondDict[key]
    for key in secondDict.keys():
        if testVec[featIndex] == key: # 寻找到相对应的分支，并进入该分支
            if type(secondDict[key]).__name__ == 'dict':   #如果是字典，继续递归
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key] #获得该叶子节点的标签
    return classLabel #返回该标签

'''
3.3.2 决策树的存储
    使用python中的pickle序列化对象，可以在磁盘上保存对象，并在需要的时候读取出来
    
'''
#3-9.使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

