# autor: zhumenger
import trees
import treePlotter
myDat, labels = trees.createDataSet()
print(labels)
myTree = treePlotter.retrieveTree(0)
print(myTree)

print(trees.classify(myTree, labels, [1, 1]))

trees.storeTree(myTree, 'classifierStorage.txt')
print(trees.grabTree('classifierStorage.txt'))