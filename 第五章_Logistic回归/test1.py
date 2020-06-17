# autor: zhumenger
import logRegres
from numpy import *
dataArr, labelMat = logRegres.loadDataSet()
print(logRegres.gradAscent(dataArr, labelMat))
weigths = logRegres.stocGradAscent1(array(dataArr), labelMat)
print(logRegres.plotBestFit(weigths))