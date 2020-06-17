# autor: zhumenger
import trees
myDat, lables = trees.createDataSet()
print(myDat)
print(lables)
print(trees.calcShannonEnt(myDat))#返回期望值, 期望值越高，则混合的数据也越多

myDat[0][-1] = 'maybe'
print(trees.calcShannonEnt(myDat))

#测试splitDataSet()
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))

trees.chooseBestFeatureToSplit(myDat)

#寻找最好的划分方式
print(trees.chooseBestFeatureToSplit(myDat)) #得到按照第 0 个特征值进行划分的结果最好

#3-4：
print(trees.createTree(myDat, lables))