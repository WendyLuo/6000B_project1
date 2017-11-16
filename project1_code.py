# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#read csv data
train_data = pd.read_csv('traindata.csv', header=None)
train_label = pd.read_csv('trainlabel.csv', header=None)

test_data = pd.read_csv('testdata.csv', header=None)
#convert data type to array
train = np.array(train_data)
label = np.array(train_label.iloc[:, 0])
test = np.array(test_data)
result = [] #to store the final result

#preprocessing data by using three different methods
xtrain_trans = preprocessing.QuantileTransformer(random_state=0).fit_transform(train)
xtrain_normal = preprocessing.normalize(train, norm='l2')
xtrain_scale = preprocessing.scale(train)

clfExtra = ExtraTreesClassifier(n_estimators=175)

#accuracy of extremely randomized trees
value_Extra2 = cross_val_score(clfExtra, xtrain_trans, label, cv=10, scoring='accuracy')
print 'The accuracy of ExtraTreesClassifier is: ', value_Extra2.mean()

#accuracy of extremely randomized trees with quantileTransformer
value_Extra1 = cross_val_score(clfExtra, xtrain_trans, label, cv=10, scoring='accuracy')
print 'The accuracy of QuantileExtraTreesClassifier is: ', value_Extra1.mean()

#accuracy of extremely randomized trees with normalize function
value_Extra3 = cross_val_score(clfExtra, xtrain_normal, label, cv=10, scoring='accuracy')
print 'The accuracy of NormalExtraTreesClassifier is: ', value_Extra3.mean()

#accuracy of extremely randomized trees with scale function
value_Extra4 = cross_val_score(clfExtra, xtrain_scale, label, cv=10, scoring='accuracy')
print 'The accuracy of ScaleExtraTreesClassifier is: ', value_Extra4.mean()

#accuracy of decision tree with quantileTransformer
clfDecision = tree.DecisionTreeClassifier()
value_Dec = cross_val_score(clfDecision, xtrain_trans, label, cv=10, scoring='accuracy')
print 'The accuracy of DecisionTreeClassifier is: ', value_Dec.mean()

#accuracy of random forest with quantileTransformer
clfRandom = RandomForestClassifier(n_estimators=10)
value_Ran = cross_val_score(clfRandom, xtrain_trans, label, cv=10, scoring='accuracy')
print 'The accuracy of RandomForestClassifier is: ', value_Ran.mean()

#accuracy of SVM with quantileTransformer
clfSVM = SVC(kernel='rbf', probability=True)
value_SVM = cross_val_score(clfSVM, xtrain_trans, label, cv=10, scoring='accuracy')
print 'The accuracy of SVMClassifier is: ', value_SVM.mean()

#fit the preprocessing data and get the prediction of test data
clfExtra.fit(xtrain_trans, label)
res = clfExtra.predict(test)
result = pd.DataFrame(res)

result.to_csv('project1_20461341.csv', index=False, header=False)
