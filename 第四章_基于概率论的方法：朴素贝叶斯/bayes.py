# autor: zhumenger
from numpy import *

# 4.1 词表到向量的转换函数
#构建一个快速过滤器，屏蔽掉侮辱性言论
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not 1代表侮辱性单词，0代表正常言论
    return postingList,classVec

# 用于制作词汇表
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的集合，用于记录所有出现过的词汇
    for document in dataSet:
        vocabSet = vocabSet | set(document) #求俩个集合的并集
    return list(vocabSet)

# 用来记录词汇表中的单词是否在输入参数中出现过
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) # 创建一个集合，0表示单词没有出现过，1表示出现过
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 求出每个单词是侮辱性单词还是正常单词的概率，以及侮辱性言论总的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 求出侮辱性言论的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #分别用来记录每个单词是侮辱性单词还是正常词汇的概率
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #每个侮辱单词出现的次数/总的单词出现的次数
    p0Vect = log(p0Num/p0Denom)          # 为防止小数溢出，这里取log
    return p0Vect,p1Vect,pAbusive

# 4-3 朴素贝叶斯分类函数
# 待分类的向量， 以及上面函数求出的3个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 前面代码的整合，求一个待分类的向量的类别
def testingNB():
    listOPosts,listClasses = loadDataSet() # 得到样本数据
    myVocabList = createVocabList(listOPosts) # 得到词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc)) # 将每个语句出现的词汇记录下来
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) # 得到每种情况的概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) # 得到输入样例单词的出现情况
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)) # 判断类别
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# 将字符串解析为字符串列表
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 对垃圾邮件分类器进行自动化处理
def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):        # 分别打开邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList) # 将字符串列表添加进去
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read()) # 将文本解析为字符串列表
        docList.append(wordList)  # 将字符串列表添加进去
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 得到词汇表
    trainingSet = list(range(50))
    testSet = []  # create test set
    for i in range(10):             # 随机抽取10个文件作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []   # 剩下的40份文件作为训练集
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet: # 遍历测试集，
        wordVector = setOfWords2Vec(vocabList, docList[docIndex]) # 得到哪些单词出现过
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]: # 判断计算出来的结果是否与其真实结果一致
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet)) # 输出错误率
    # return vocabList,fullText

