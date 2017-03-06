# coding=utf-8


from numpy import *
import operator


"""
  数据集合
  1\ k 临近算法
  优点：
    （1）
  缺陷：
    （1）
"""
def  createDataSet():
    group = array( [ [ 1.0, 1.1 ], [ 1.0, 1.2 ] , [ 0, 0.3 ], [ 0, 0.1 ] ] );
    labels = [ 'A' , 'A' , 'B' , 'B' ]
    return  group, labels



"""
 1、已知数据 对应的 已分类
 2、inX => 输入 需要 预测的数据
 3、k 个类 数据 进行比较

"""
def classify0(dataSet , labels , inX  , k ):
    dataSetSize = dataSet.shape[0]; # 矩阵, 行数
    if( k > dataSetSize ) : k = dataSetSize

    #算距离
    inMat =  tile( inX, ( dataSetSize, 1 ) ); # 复制（ row ,col ）
    diffMat =  inMat - dataSet;
    sqDiffMat =  diffMat**2;
    sqDistance = sqDiffMat.sum(1);
    distance = sqDistance**0.5 ;
    sortedDistance = distance.argsort(); # 数组值从小到大的排序 索引值

    classCount = {};
    for i in range( k ): #
        j = sortedDistance[ i ]; # 距离最短的 行 数据，
        label = labels[ j ]; #
        classCount[ label ] = classCount.get( label , 0 ) + 1; # 命中 标签 权值 ++

    # 对 命中标签的字典排序，获取 最大权值的 标签
    sortedList = sorted( classCount.iteritems() , key = operator.itemgetter(1), reverse= True  ); # 元组列表， 比较元祖的 第二个值，
    return sortedList[0][ 0 ]; # 列表 第一个 原素的 元组的 第一个元素（ 类标签 ）


"""
 1\ 文件中 读取数据, 构造矩阵
 2、
"""
def file2Matrix( filename ):
    fr = open( filename , 'r' );
    lines = fr.readlines();
    # 初始化 矩阵  n 行， 3 列
    mat = zeros( ( len( lines ), 3 ) );
    classlabels = [ ]; # 分类标签

    index = 0;
    for line in lines :
        line = line.strip();
        list = line.split( "\t"); #
        # 第 index 行的数据 和 标签
        mat[ index , :] = list[ 0:3 ];
        classlabels.append( list[ -1 ] );
        index += 1;

    return mat, classlabels;


"""
  1\ 将数值 取值范围 变成 0-1 之间；
"""

def autoNorm( dataSet ):
    minV = dataSet.min( 0 ); # 每列最小数据, 构成 n 列
    maxV = dataSet.max( 0 );
    ranges = maxV - minV ;

    rowAndCol = shape( dataSet ); # row col 元组，
    row = rowAndCol[0];
    resultSet = zeros( rowAndCol ); # 构造一个 与 dataSet 等行等列的 矩阵

    resultSet = dataSet - tile( minV , ( row , 1 ) ); # 矩阵【 i，j 】 - [ i , j ]（最小值）
    resultSet = resultSet/ tile( ranges , ( row , 1 ) );

    return  resultSet, ranges , minV ;



def datingClassTest():
    hoRatio = 0.1   #hold out 10%
    datingDataMat,datingLabels = file2Matrix('/Users/wanghui/PycharmProjects/FirstPython/file/datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0] # 行数
    numTestVecs = int(m*hoRatio) # 一半 测试行数，
    errorCount = 0.0
    for i in range(numTestVecs):
        # dataSet 后一半 矩阵，样本，
        # labels ,后一半 标签分类，
        #  inX  , 测试数据， 前一半数据
        # k ，3
        classifierResult = classify0(  normMat[ numTestVecs:m , : ] , datingLabels[ numTestVecs:m ] , normMat[ i, : ] ,  4 )
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))



