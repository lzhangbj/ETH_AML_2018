
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
#IO
trainXdf=pd.read_csv('X_train.csv')
trainYdf=pd.read_csv('y_train.csv')
testXdf=pd.read_csv('X_test.csv')
trainX=trainXdf[[i for i in trainXdf.columns if i not in ['id']]]
trainY=trainYdf[[i for i in trainYdf.columns if i not in ['id']]]
#i will preprocess the columns
X=trainX
y=trainY['y'].values


# In[ ]:


import sklearn.model_selection
import sklearn.metrics
import autosklearn.regression


# In[ ]:


#feature selection
from scipy.stats import pearsonr
features = trainX.columns.tolist()
target =  trainY.columns[0] #'y'

pearsonSelectThreshold=0.2

correlations = {}
for f in features:
    nonEmptyRows=~trainX[f].isnull()
    x1 = trainX[f].values[nonEmptyRows]
    x2 = trainY[target].values[nonEmptyRows]
    correlations[f] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
featureImpotance=data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
#filter columns that are irrelevant, such as contant columns, zero columns that may crash everything
gdFeature=featureImpotance[np.abs(featureImpotance.Value)>pearsonSelectThreshold]
print('features with higher correlation with y: ')
print(gdFeature.head())
gdFeatureName=gdFeature.index.tolist()


# In[ ]:


#create processed matrix for training and testing
X=X[gdFeatureName]
testX=testXdf[gdFeatureName]


# In[ ]:


feature_types = (['numerical'] *  len(X.columns))
X_train, X_test, y_train, y_test =     sklearn.model_selection.train_test_split(X, y)
print('Start!')
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=360, #allow 6 minutes to train the ensemble
    per_run_time_limit=30, #allow half minutes to train each model

)
#this find the ensemble of models 
automl.fit(X_train, y_train, dataset_name='task1',
           feat_type=feature_types)


print(automl.show_models())
#print score, usually get a better score in publicboard, i d k y
predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
print('Training Finished!')


# In[ ]:


#look up the models inside 
ensemble=automl.get_models_with_weights()
print(ensemble)


# In[ ]:


r2=[]
n=5
for i in range(n):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
    #this line does nothing except retraining the model
    automl.refit(X_train, y_train)
    #print score, usually get a better score in publicboard, i d k y
    predictions = automl.predict(X_test)
    r2score=sklearn.metrics.r2_score(y_test, predictions)
    print("R2 score:", r2score)
    r2.append(r2score)
print("Average R2 score over monte carlo cross validation (n={}):\t{}".format(n,np.mean(r2)))


# In[ ]:


#retrain the model and predict the test set
automl.refit(X, y)
predictions = automl.predict(testX)


# In[ ]:


#warp up the result for the test set
result=pd.DataFrame()
result['id']=testXdf['id']
result['y']= predictions
result.head()


# In[ ]:


#save, good luck
# result.to_csv('trial1999.csv')

