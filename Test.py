# coding=utf-8
import os
import time;

import operator
import matplotlib
import matplotlib.pyplot
from numpy.ma import multiply

from  Alg import bayes
from matplotlib import pyplot
from  numpy import random , array, tile, zeros, shape, mat, transpose, asarray
from Alg import KNN;
from Alg import DecisionTree;

from Emp import Emp;


"""
"""

# postingList,classVec = bayes.loadDataSet();
# myWordList = bayes.createVocabList( postingList );
# print "wordlist :",myWordList
#
#
# returnVec = bayes.testNB(  )
#
# print returnVec;

# matrix = mat( [ [ 1,2,3,4 ] , [ 5,6,7,8 ] ]);
# print matrix;
#
# t = transpose( matrix );
# print  t;


a = mat(
    [
     [ 1 ],
     [ 1 ]
    ]
);
label = mat(
    [ [ -1 , 2 ,3 ],
      [ 1 ,4, 5   ]
    ]
);

v = a-label;
print  v;

#
# ala = asarray( a ) ;
# print ala;
#
# print multiply(3,4);


# mat = zeros( ( 10, 3) );
# classCount = {
#     'E': 8,
#     'F': 0,
#     'A': 5,
#     'C': 6,
#     'D': 1,
#     'B': 3,
#
# };
#
# print zeros( (2,5 ) );


# resortedList = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True );
# print resortedList;
# print resortedList[0][0];
#
# tree = {
#          'no surfing':
#              {
#                  0:'no',
#                  1 :
#                       { 'flippers':
#                             { 0:'no',
#                               1:'yes'
#                             }
#                       }
#              }
#         };
# print tree['no surfing'].keys();

"""
 f1(age)    f2(preScript)    f3(astigmatic)     f4(tearRate)         decision
young	    myope	          no	             reduced	        no lenses
young	    myope	          no	             normal	            soft
old 	    myope	          yes	             reduced	        no lenses

"""
# fr  = open("/Users/wanghui/PycharmProjects/FirstPython/file/lenses.txt");
# data = [  lines.split("\t") for lines in fr.readlines( ) ];
# features = [ 'age', 'preScript' , 'astigmatic' , 'tearRate' ];
# tree = DecisionTree.createTree( data , features );
# print  tree;
# DecisionTree.storeTree( tree , os.curdir+"/file/tree.txt");
#
# cacheTree = DecisionTree.grabTree( os.curdir+"/file/tree.txt");
# print "cachs", cacheTree;
#
# featLables = [ 'preScript', 'astigmatic' , 'tearRate' , 'age' ];
# testVes =  [ 'hyper', 'no' , 'normal' , 'presbyopic' ]
#    # ['myope', 'no',  'normal',  'young'],
#
# print  DecisionTree.classify( cacheTree , featLables , testVes );




#
#
# fileName = "path/%s" % "file";
# print  fileName;
#KNN.datingClassTest();
#
#
#
# mat, classlabels  = KNN.file2Matrix("/Users/wanghui/PycharmProjects/FirstPython/file/datingTestSet.txt");
#
# # resultSet, ranges, minV = KNN.autoNorm( mat );
#
# row,col = shape( mat );
# print row, col , shape( mat )[ 1 ];
# print mat;
#
# print classlabels;
#
# resultSet, ranges , minV  = KNN.autoNorm( mat );
#
# print resultSet;

# fig = pyplot.figure();
# ax = fig.add_subplot( 111 );
# ax.scatter( mat[ :, 0 ] , mat[ :,2 ]  );
# pyplot.show();



# t = array(
#      [
#     [ 3,5,0 ],
#     [ 6,4,7 ],
#     [ 1,4,3 ]
#      ] );
#
# print t.min(1);
#
# mat[ 1, : ] = t[ 0:3];
# print mat;
#
# dataSet = \
#     [
#         [ 1, 'Y'],
#         [ 2, 'N'],
#         [1, 'Y'],
#         [2, 'Y'],
#         [1, 'Y'],
#         [2, 'Y'],
#         [1, 'Y'],
#         [2, 'N'],
#         [2, 'M'],
#         [2, 'K'],
#     ];
#
# print DecisionTree.calculateShannonEnt( dataSet );

#
# varlist = [ 'te', 78 , '11.02', int(90.03)]
# print range( len( varlist) -1 );
# tuplevar = ( 3,45,89 ) ;
# var_dic = { 'g': 1, '2': 5 ,  'e3': 4 , 6:0 }
# for item in  var_dic.iteritems():
#     print item[0], item[1]
#
# operatorF = operator.itemgetter( 0);
# print operatorF( var_dic.values() );
#
# iter = var_dic.iteritems();
#
# var_dic = sorted( var_dic.iteritems() , key= operator.itemgetter(1), reverse=True );
# print var_dic;
#
# varlist = array(
#     [
#         [2, 0 , 10,11  ],
#         [12,213, 0 ,22 ],
#         [122,23, 89,15 ],
#     ] );
# print varlist.shape;
# #
# inX = [ 2.3, 0.5 , 0.8 , 9 , 9 ];
# print  set( inX );
#
# print inX[ : 2 ], inX[ 2+1 :];
#
# dataSetSize = varlist.shape;
# print "dataSetSize= row:",dataSetSize[0]," col=", dataSetSize[1] ;
# inXarray = tile( inX , ( 3 , 1 ) );
# print 'in array = ', inXarray;
#
# diffMat = inXarray-varlist;
# sqlMat =  diffMat**2;
# total = sqlMat.sum(1);
# dis = total**0.5;
# print 'total=',total;
# print "dis=",dis;
# print "dis.argsort =",dis.argsort();
# print diffMat;
# print diffMat**2;

"""
print  "2=", tile( varlist , 2 );
print "2*3 = ",tile( varlist , (2, 3 ) );
print "2*3*4 = ",tile( varlist , (2, 3, 4 ) );

varlist = array(
    [
        [2, 3, 0, 1,2, 3, 0, 1,2, 3, 0, 1,2, 3, 0, 1],
        [12, 13, 0, 22],
        [22, 23, 0, 15],

        [2, 3, 0, 1],
        [12, 13, 0, 22],
        [22, 23, 0, 15],

        [2, 3, 0, 1],
        [12, 13, 0, 22],
        [22, 23, 0, 15],
    ],

);



print  random.random( 4 );
print  random.rand(4,4)
print varlist[0]
print varlist[0:3];
print varlist[-3:9];
print var_dic['two'];
print var_dic.keys();
print var_dic.values();

a = 2; b=8;
print 'a**b=',a**b;
print  var_dic.get( 'twoo', 'we');

c = 01;
d = 11;

print 'c&d=',c & d;

a = 0;
if( a and b ):
    print "true";
else:
    if( a or b ):
        print "a or b :", b ;
    elif not ( a or b ):
        print " a not b", a,b;
    else:
        print "else";


a = 'one';
if( a in var_dic ):
    print "a in dic = ",var_dic[ a ];


a = "2"; b = "1";
print 'a is  b =', a is b ;

arrays = [ 12, 57 ,89, 0 ];

even = [ ] ;
odd = [ ];
while( len( arrays ) > 0 ):
    n = arrays.pop();
    if( n % 2 ==0 ):
        even.append( n );
    else:
        odd.append( n );

print even,odd;
print odd;


str = 'hello word';
print str[ : len(str) ];
print 'add'+ str[:];
print  str[2:-2]+"word22";



print time.time();
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ;



def printMe( str ):
   print "打印放大";
   return True;

def changeMe(  myList ):
    myList+=( myList );
    return myList;





print  printMe( 1 ) == False ;


print 'varlist = ',varlist;
changeMe( varlist );
print 'varlist = ',varlist;


print 'a = ',a;
print 'a changme =',changeMe( int(a) );
print 'a = ',a;



emp1 = Emp( 'test', 20 );
print emp1.getSalary();

emp2 = Emp( 'test', 20 );
print emp2.getSalary();

print 'yu gong ', Emp.empCount;

print  emp1 is emp2;


print range( 5 );


"""





