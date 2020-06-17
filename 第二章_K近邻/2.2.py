# autor: zhumenger
from numpy import *
import KNN
datingDataMat, datingLables = KNN.file2matrix('datingTestSet2.txt')
'''print(datingDataMat)
print(datingLables[0:20])'''

#利用matplotlib库图形化展示数据
# 制作原始数据的散点图
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#scatter支持个性化标记散点图上的点  [:,index]获得下标为index + 1的整体数据元素

ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLables), 15.0 * array(datingLables))
plt.show()

normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
'''print(normMat)
print(ranges)
print(minVals)'''

KNN.datingClassTest()

KNN.classifyPerson()