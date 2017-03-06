# coding=utf-8

"""
 朴素贝叶斯
"""
from math import log

from numpy import ones


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec



"""
  从 数据源 中 构建单词 集合
"""
def createVocabList( dataSet ):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)


"""
   vocabList = [ 'my', 'dog' ]
   inputSet = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
   returnVec = [  1 ,  1     ],
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec


"""
   postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
"""





"""
 朴素贝叶斯 分类算法 区分文本分类， 构造特征矩阵
 (1) 构建 单词 库
 （2）根据单词库，构造特征矩阵
"""
def createFeatureMat( dataSet ):
    # 数据源 所有的 单词 矩阵 【 'w','f', 'g' 】
    myWordList = createVocabList( dataSet );
    # 构建 训练 矩阵
    trainMat = [];
    for postDoc in dataSet:
        fVec = setOfWords2Vec(myWordList, postDoc);  # 待测 数据 与 所有 单词列表相比较的， 特征矩阵
        trainMat.append(fVec);

    return trainMat,myWordList;


"""
朴素贝叶斯 分类算法 区分文本分类
(1) trainMatrix 特征矩阵，
（2）trainCategory 对应的 文本类型
"""
def trainNB0(trainMatrix, trainCategory):
    trainLen = len(trainMatrix)  # 总样本数
    words = len(trainMatrix[0])  # 总特征数， 为 单词库 总数

    abusiveProb = sum(trainCategory) / float(trainLen)   # 是  的概率
    #
    p0Num = ones( words );  p1Num = ones( words );
    p0Denom = 2.0;p1Denom = 2.0
    for i in  range( trainLen ):
        if trainCategory[ i ] == 1: #
            p1Num += trainMatrix[i]
            p1Denom += sum( trainMatrix[ i] )
        else: # 0
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, abusiveProb



"""
  朴素贝叶斯 分类算法 => 区分 是1 非0
  Native bayes
"""
def classifyNB( vec2Classify , p1Vec , p0Vec, pClass1 ):


    return [ ];


def testNB():

    listOpts , classVec = loadDataSet();
    # 数据源 所有的 单词 矩阵 【 'w','f', 'g' 】
    trainMat, myWordList = createFeatureMat( listOpts );
    print trainMat, myWordList;















