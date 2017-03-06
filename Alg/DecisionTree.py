# coding=utf-8


import math
from matplotlib import pyplot as plt
import operator
from numpy import array

"""
 决策树 算法
 明确结论的算法

    实现（1）ID3算法
  从不熟悉的数据中提取出一系列规则

  怎么提取？
  （1）信息熵、 信息增益

  标称型数据


"""

# 计算信息熵
"""
   dataSet = [    #纬度1，2，3，4  结论L1
                 [ 1,2,3,4, 'L1' ] ,
                 [ 1,2,3,4, 'L2' ] ,
                 [ 1,2,3,4, 'L3' ] ,
                 [ 1,2,3,4, 'L4' ] ,
             ]
"""


def calculateShannonEnt(dataSet):
    #
    #
    lenth = len(dataSet);  # 数据集合 行数
    labelCount = {};  # 结论 加权 字典
    for data in dataSet:
        lastCol = data[-1];  # 结论
        if lastCol not in labelCount.keys():
            labelCount[lastCol] = 0;
        labelCount[lastCol] += 1;  # 该结论 权重 ++

    # 计算 香浓熵
    shannonEnt = 0.0;
    for item in labelCount.iteritems():
        key = item[0];  # 结论
        value = float(item[1]);  # 元祖对，比重数
        prob = value / lenth;  # 结论A 概率
        shannonEnt -= prob * math.log(prob, 2);
    return shannonEnt;


"""
 划分数据集合
  1、 dataSet 被划分的 数据集合
  2、 axis 选取的 纬度轴
  3、 纬度轴 所对应的值

   dataSet = [    #纬度1，2，3，4  结论L1

                 [ 1,2,3,4,  'L1' ] ,
                 [ 12,2,3,4, 'L2' ] ,
                 [ 23,2,3,4, 'L3' ] ,
                 [ 1,2,3,4,  'L4' ] ,
             ]

"""


def splitDataSet(dataSet, axis, value):
    splitedDataSet = [];
    for data in dataSet:
        if (data[axis] == value):
            reducedV = data[: axis];
            reducedV.extend(data[axis + 1:]);  # 不包括 axis 轴数据
            splitedDataSet.append(reducedV)
    return splitedDataSet;


"""
  选取 一个 最好的 特征 划分数据集
   （1）怎样判断 一个 特征是 最好的 特征 ？ 分类熵值，与 原始熵值 变化最大，  信息增益最大

"""


def chooseBestFeature(dataSet):
    bestFeature = -1;
    # 原始 香浓熵
    orgShannEont = calculateShannonEnt(dataSet);

    # 特征值 个数
    features = len(dataSet[0]) - 1;
    for f in range(features):  # 特征 f = 0，1，2，3，4 ，
        featureToValues = [ v[f] for v in dataSet ];  # 特征 f 所有的 取值
        featureToValues = set(featureToValues);
        # 计算 特征 f 的 分类; E( 熵值 | 特征f ) =
        newShannEnot = 0.0;
        bestInfoGain = 0.0;
        for fv in featureToValues:
            splitSet = splitDataSet(dataSet, f, fv);  #
            prob = float(len(splitSet)) / float(len(dataSet));  # 特征f 值是 fv 的 分组所占比重
            newShannEnot += prob * calculateShannonEnt(splitSet);

        # 信息增益；越大，信息越有序， 熵值越小越好
        infoGain = orgShannEont - newShannEnot;
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain;  # 保存 信息增益最大值，最好的特征f
            bestFeature = f;

    # 访问所有的 特征划分好,
    return bestFeature;


"""
  根据 多数表决 决定叶子节点的 分类
"""


def majorityCnt(classList):
    classCount = {};
    for vote in classList:
        if vote not in classCount.keys: classCount[vote] = 0;
        classCount[vote] += 1;

    # 词典排序 返回的 是 =》元祖列表
    resortedList = sorted(classCount.iteritems(), key=operator.getitem(1), reverse=True);  # 元祖对 , 比较的是 （k,v） v ,
    return resortedList[0][0];  # 返回最大值得 k => 特征值


"""
 构造 决策树
  1、结束条件：

"""


def createTree(dataSet, labels):
    classList = [data[-1] for data in dataSet];
    if classList.count(classList[0]) == len(classList): return classList[0];  # 都是 同一类 ，只有 一个叶子节点，

    # 选取 叶子节点==》 特征值
    bestFeature = chooseBestFeature( dataSet );
    bestFeatureToValues = set( [  v[ bestFeature ] for v in dataSet ] );
    bestFeatureLabels = labels[ bestFeature ];

    del( labels[ bestFeature ] ); # 特征值已使用

    # 保存 当前的 树
    tree = {  bestFeatureLabels :{} }; # 词典
    for fv in bestFeatureToValues:
        # 分类
        splitedData = splitDataSet(dataSet, bestFeature, fv ); # 新的数据集

        # 新的标签
        subLabels = labels[ : ];
        tree[ bestFeatureLabels ][ fv ] = createTree( splitedData , subLabels ); # f 节点下 的 子树

    return  tree;


"""
1、使用决策树， 进行分类

 testVec   = [  1,  0  ,1  , 0  ]
 featLabels= [ f1, f2 , f3 , f4 ]
"""

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]

    featIndex = featLabels.index(firstStr); # 0
    key = testVec[featIndex]; # 1

    valueOfFeat = secondDict[key]
    # 如果还有 子节点（ 树 ）
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat

    return classLabel


"""
 存储
"""


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

"""
  测试

"""









