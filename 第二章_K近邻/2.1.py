# autor: zhumenger
import KNN
group, labels = KNN.createDataSet()
print(KNN.classify([0, 0], group, labels, 3))