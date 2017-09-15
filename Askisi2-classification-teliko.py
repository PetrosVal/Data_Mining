
# coding: utf-8

# In[376]:

import pandas
#import operator
import collections

from sklearn import preprocessing
from sklearn import grid_search
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve


# In[377]:

traindf = pandas.read_csv('train.tsv', sep='\t')
testdf = pandas.read_csv('test.tsv', sep='\t')

le = preprocessing.LabelEncoder()

le.fit(traindf.Attribute1)
traindf["Attribute1"]=le.transform(traindf.Attribute1)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute3)
traindf["Attribute3"]=le.transform(traindf.Attribute3)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute4)
traindf["Attribute4"]=le.transform(traindf.Attribute4)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute6)
traindf["Attribute6"]=le.transform(traindf.Attribute6)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute7)
traindf["Attribute7"]=le.transform(traindf.Attribute7)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute9)
traindf["Attribute9"]=le.transform(traindf.Attribute9)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute10)
traindf["Attribute10"]=le.transform(traindf.Attribute10)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute12)
traindf["Attribute12"]=le.transform(traindf.Attribute12)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute14)
traindf["Attribute14"]=le.transform(traindf.Attribute14)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute15)
traindf["Attribute15"]=le.transform(traindf.Attribute15)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute17)
traindf["Attribute17"]=le.transform(traindf.Attribute17)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute19)
traindf["Attribute19"]=le.transform(traindf.Attribute19)

le = preprocessing.LabelEncoder()
le.fit(traindf.Attribute20)
traindf["Attribute20"]=le.transform(traindf.Attribute20)
#print traindf


# In[378]:

Label=traindf["Label"]
#print Label
traindf.pop("Label")
traindf.pop("Id")
#print traindf


# In[379]:

le = preprocessing.LabelEncoder()

le.fit(testdf.Attribute1)
testdf["Attribute1"]=le.transform(testdf.Attribute1)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute3)
testdf["Attribute3"]=le.transform(testdf.Attribute3)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute4)
testdf["Attribute4"]=le.transform(testdf.Attribute4)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute6)
testdf["Attribute6"]=le.transform(testdf.Attribute6)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute7)
testdf["Attribute7"]=le.transform(testdf.Attribute7)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute9)
testdf["Attribute9"]=le.transform(testdf.Attribute9)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute10)
testdf["Attribute10"]=le.transform(testdf.Attribute10)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute12)
testdf["Attribute12"]=le.transform(testdf.Attribute12)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute14)
testdf["Attribute14"]=le.transform(testdf.Attribute14)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute15)
testdf["Attribute15"]=le.transform(testdf.Attribute15)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute17)
testdf["Attribute17"]=le.transform(testdf.Attribute17)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute19)
testdf["Attribute19"]=le.transform(testdf.Attribute19)

le = preprocessing.LabelEncoder()
le.fit(testdf.Attribute20)
testdf["Attribute20"]=le.transform(testdf.Attribute20)
#print testdf


# In[380]:

#testdf.pop("Id")
id_predicts=testdf.pop("Id")
#print id_predicts


# In[381]:

#svd = TruncatedSVD(n_components=3)


# In[382]:

import numpy as np
X=np.array(traindf)
y=np.array(Label)
z=np.array(testdf)
#print X
#print y
#print z
clf = SVC()
clf.fit(X, y,sample_weight=None) 
test_predicts=clf.predict(z)
#print "test_predicts"
#print test_predicts


# In[383]:

clf=RandomForestClassifier(n_estimators=10)
clf.fit(X, y) 
test_predicts=clf.predict(z)
test_predicts_rf=test_predicts
#print "test_predicts"
#print test_predicts


# In[384]:

clf=MultinomialNB()
clf.fit(X, y) 
test_predicts=clf.predict(z)
#print "test_predicts"
#print test_predicts


# In[385]:

evaluation_result=[]
from sklearn import cross_validation


# In[386]:

scores=cross_validation.cross_val_score(SVC(),X,y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
evaluation_result.append(scores.mean())
#print evaluation_result


# In[387]:

scores=cross_validation.cross_val_score(RandomForestClassifier(n_estimators=10),X,y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
evaluation_result.append(scores.mean())
#print evaluation_result


# In[388]:

scores=cross_validation.cross_val_score(MultinomialNB(),X,y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
evaluation_result.append(scores.mean())
#print evaluation_result


# In[389]:

one = "Accuracy"
evaluate = np.array(evaluation_result) 
evaluation = [('Statistic Measure',[one]),('SVC', [evaluate[0]]), ('RandomForests', [evaluate[1]]), ('Naive Bayes', [evaluate[2]])]
print evaluation
df_evaluation = pandas.DataFrame.from_items(evaluation)
#df_evaluation 


# In[390]:

df_evaluation.to_csv('EvaluationMetric_10fold.csv', sep='\t')


# In[391]:

#df_predicted = pandas.DataFrame({'Client_ID':id_predicts,'Predicted_Label':test_predicts_rf})
#df_predicted
#df_predicted.to_csv('testSet_Predictions_dok.csv',sep='\t')


# In[392]:

#id_predicts
#test_predicts_rf
#dimiourgeitai dataframe gia ton RandomForests pou paraousiazei thn kalyterh akriveia->73%-75%
test_predicts_char=[]
mikos=len(test_predicts_rf)
for x in range(mikos):
    if(test_predicts_rf[x]==1):
        test_predicts_char.append('Good')
    else:
        test_predicts_char.append('Bad')
#print test_predicts_rf        
#print test_predicts_char         
df_predicted = pandas.DataFrame({'Client_ID':id_predicts,'Predicted_Label':test_predicts_char})
df_predicted.to_csv('testSet_Predictions.csv',sep='\t')


# In[ ]:



