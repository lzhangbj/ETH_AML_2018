
# coding: utf-8


import numpy
import pandas as pd
import matplotlib.pyplot as plt

#preprocessing tool
from biosppy.signals import ecg

#machine learning tools
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler



clf=XGBClassifier(silent=0,objective='multi:softmax')


numpy.random.seed(0)




#IO
path=''
trainXdf=pd.read_csv(path+'X_train.csv',index_col='id')
trainYdf=pd.read_csv(path+'y_train.csv',index_col='id')
testXdf=pd.read_csv(path+'X_test.csv',index_col='id')
#i will preprocess the columns
X=trainXdf
y=trainYdf['y'].values



#data preprocessing
def process(X):
    Fs=300
    templates={}
    templates_var={}
    the_heartbeats={}
    data=X
    nsamples=data.shape[0]
    for i in range(data.shape[0]):
        if not i%50: #print message every 50 samples
            print('preprocessed {} / {} samples'.format(i,nsamples))
        signal1=data.iloc[i].dropna().values
        out = ecg.ecg(signal=signal1, sampling_rate=Fs, show=False)
        templates[data.iloc[i].name]=out['templates'].mean(0)
        templates_var[data.iloc[i].name]=out['templates'].var(0)
        the_heartbeats[data.iloc[i].name]=out['heart_rate']

    #heartbeat teamplate's mean and variance at each time step
    templatesDF=pd.DataFrame(templates).T
    templatesDF.index=data.index
    templates_varDF=pd.DataFrame(templates_var).T
    templates_varDF.index=data.index

    #heartbeat rate's mean and variance
    hbs=[[heartbeats.mean(), heartbeats.var()] for (i,heartbeats) in the_heartbeats.items()]
    hbsDF=pd.DataFrame(hbs)
    hbsDF.columns=['heartbeats mean','heartbeats variance']
    hbsDF.index=data.index

    X_=pd.merge(templatesDF,templates_varDF, left_index=True, right_index=True)
    X_=pd.merge(hbsDF, X_, left_index=True, right_index=True)
    return X_



X_=process(X)
idx=X_.dropna().index #some entry being nan in heartbeat rate, only 4 of them anyway, i didn't bother much


#one pass evaluation for model, with oversampling
X_train, X_test, y_train, y_test = train_test_split(X_.iloc[idx], y[idx])
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
clf.fit(X_resampled, y_resampled, eval_metric=f1_score)
pred=clf.predict(X_test.values)
scores=f1_score(y_test, pred, average='micro')
print(scores)



#cross validation, got 0.7x for cv=3, 5, 10.
#did not apply oversampling,
#adding a pipeline in cross validation is doable but i am too lazy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_.iloc[idx], y[idx], cv=3, scoring='f1_micro',verbose=1000)
print(scores)



#cv results are not bad, retrain the model and predict the test set
clf.fit(X_.iloc[idx], y[idx], eval_metric=f1_score)

#see if the model fits the training set well, prevent unknown bugs
own_pred=clf.predict(X_.iloc[idx])
own_score=f1_score(y[idx],own_pred,average='micro')
print(own_score)
#...got 0.8x, seemingly does not overfit, nice!


#preprocess the test set and predict labels
testX_df=process(testXdf)
y_pred=clf.predict(testX_df)
y_pred



#clean up for exporting the result
result=pd.DataFrame()
result['id']=testX_df.index
result['y']= y_pred
result.head()
#last glimpse to the predict result, looks like this-> [ |.'. ], reasonable
result.hist()

#save the result to disk
result.to_csv('trial1.csv',index=False)

#submitted with public score 0.76318> hard benchmark=0.73564
