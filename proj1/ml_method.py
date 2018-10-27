import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression,ElasticNet,LogisticRegression,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.impute import SimpleImputer

import autosklearn.regression

import warnings
warnings.filterwarnings("ignore")

EPSILON = 1e-14

def R_score(y_true, y_pred):
    SS_res =  np.sum(np.square( y_true-y_pred ))
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + EPSILON) )

TRAINING = False
TESTING  = True

DECISIONTREE = False
RANDOMFOREST = True
XGBOOST      = True
GRADIENTBOOST= True
SVM 		 = True
KNN 		 = True
EXTRATREE 	 = True
ADABOOST 	 = True
MIXING		 = True


DecisionTreeModel = DecisionTreeRegressor()
RandomForestModel=RandomForestRegressor(n_estimators=40, random_state=8,max_features='auto',oob_score=True)
XGBoostModel=XGBRegressor(reg_alpha=0.08)
GradientBoostingModel=GradientBoostingRegressor(loss='huber', random_state=7, max_depth=6)
SVMModel=svm.SVR(kernel='sigmoid', gamma="auto_deprecated")
KNNModel=KNeighborsRegressor(100, weights="uniform", algorithm='auto')
ExtraTreeModel=ExtraTreesRegressor(random_state=8,min_samples_split=4)
AdaBoostModel=AdaBoostRegressor(n_estimators=100,learning_rate=0.2,random_state=2,loss='square')


TestModel =  autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=360, #allow 6 minutes to train the ensemble
    per_run_time_limit=30, #allow half minutes to train each model
)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')


train_data_dir 	= "data/train.npy"
dev_data_dir	= "data/dev.npy"
test_data_dir 	= "data/test.npy"

train_data = np.load(train_data_dir)
X_test = np.load(test_data_dir)
dev_data = np.load(dev_data_dir)
train_data_all = np.concatenate([train_data, dev_data], axis=0)

X_train = train_data[:,1:]
y_train = train_data[:,0]
X_dev = dev_data[:, 1:]
y_dev = dev_data[:, 0]
X_train_all = train_data_all[:, 1:]
y_train_all = train_data_all[:, 0]

train_N = X_train.shape[0]
dev_N   = X_dev.shape[0]
test_N 	= X_test.shape[0]


scaler = StandardScaler()
if TRAINING:
	# preprocess_data_mean(train_data, X_test, dev_data)

	X_train=imp_mean.fit_transform(X_train)
	X_dev = imp_mean.transform(X_dev)
	X_train = scaler.fit_transform(X_train)
	X_dev = scaler.transform(X_dev)

if TESTING:
	# preprocess_data_mean(train_data_all, X_test)
	X_train_all=imp_mean.fit_transform(X_train_all)
	X_test = imp_mean.transform(X_test)
	X_train_all = scaler.fit_transform(X_train_all)
	X_test = scaler.transform(X_test)
	

x_train_predictions = None
x_dev_predictions   = None


# TestModel =  MLPRegressor(alpha=0.01, hidden_layer_sizes=(32), max_iter=300)

if TRAINING:

	# training
	if DECISIONTREE :
		
		DecisionTreeModel.fit(X_train, y_train)
		train_predictions0 = DecisionTreeModel.predict(X_train)
		dev_predictions0=DecisionTreeModel.predict(X_dev)
		# test_predictions0=DecisionTreeModel.predict(X_test)
		score = R_score(y_dev, dev_predictions0)
		print("DecisionTree:\t{:.4f}".format(score))
		train_predictions0_t = train_predictions0.reshape((train_N,1))
		dev_predictions0_t = dev_predictions0.reshape((dev_N,1))
		# test_predictions0_t = test_predictions0.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions0_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions0_t], axis=1)
		else:
			x_train_predictions = train_predictions0_t
			x_dev_predictions = dev_predictions0_t
	if RANDOMFOREST:
		
		RandomForestModel.fit(X_train,y_train)
		train_predictions1 = RandomForestModel.predict(X_train)
		dev_predictions1=RandomForestModel.predict(X_dev)
		# test_predictions1=RandomForestModel.predict(X_test)
		score = R_score(y_dev, dev_predictions1)
		print("RandomForest:\t{:.4f}".format(score))
		train_predictions1_t = train_predictions1.reshape((train_N,1))
		dev_predictions1_t = dev_predictions1.reshape((dev_N,1))
		# test_predictions1_t = test_predictions1.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions1_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions1_t], axis=1)
		else:
			x_train_predictions = train_predictions1_t
			x_dev_predictions = dev_predictions1_t
	if XGBOOST:
		
		XGBoostModel.fit(X_train,y_train)
		train_predictions2 = XGBoostModel.predict(X_train)
		dev_predictions2=XGBoostModel.predict(X_dev)
		# test_predictions2=XGBoostModel.predict(X_test)
		score = R_score(y_dev, dev_predictions2)
		print("XGBoost:\t{:.4f}".format(score))
		train_predictions2_t = train_predictions2.reshape((train_N,1))
		dev_predictions2_t = dev_predictions2.reshape((dev_N,1))
		# test_predictions2_t = test_predictions2.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions2_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions2_t], axis=1)
		else:
			x_train_predictions = train_predictions2_t
			x_dev_predictions = dev_predictions2_t
	if GRADIENTBOOST:
		
		GradientBoostingModel.fit(X_train,y_train)
		train_predictions3 = GradientBoostingModel.predict(X_train)
		dev_predictions3=GradientBoostingModel.predict(X_dev)
		# test_predictions3=GradientBoostingModel.predict(X_test)
		score = R_score(y_dev, dev_predictions3)
		print("GBoosting:\t{:.4f}".format(score))
		train_predictions3_t = train_predictions3.reshape((train_N,1))
		dev_predictions3_t = dev_predictions3.reshape((dev_N,1))
		# test_predictions3_t = test_predictions3.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions3_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions3_t], axis=1)
		else:
			x_train_predictions = train_predictions3_t
			x_dev_predictions = dev_predictions3_t
	if SVM:
		
		SVMModel.fit(X_train,y_train)
		train_predictions4 = SVMModel.predict(X_train)
		dev_predictions4=SVMModel.predict(X_dev)
		# test_predictions4=SVMModel.predict(X_test)
		score = R_score(y_dev, dev_predictions4)
		print("SVM:\t{:.4f}".format(score))
		train_predictions4_t = train_predictions4.reshape((train_N,1))
		dev_predictions4_t = dev_predictions4.reshape((dev_N,1))
		# test_predictions4_t = test_predictions4.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions4_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions4_t], axis=1)
		else:
			x_train_predictions = train_predictions4_t
			x_dev_predictions = dev_predictions4_t
	if KNN:
		
		KNNModel.fit(X_train,y_train)
		train_predictions5 = KNNModel.predict(X_train)
		dev_predictions5=KNNModel.predict(X_dev)
		# test_predictions5=KNNModel.predict(X_test)
		score = R_score(y_dev, dev_predictions5)
		print("KNN:\t{:.4f}".format(score))
		train_predictions5_t = train_predictions5.reshape((train_N,1))
		dev_predictions5_t = dev_predictions5.reshape((dev_N,1))
		# test_predictions5_t = test_predictions5.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions5_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions5_t], axis=1)
		else:
			x_train_predictions = train_predictions5_t
			x_dev_predictions = dev_predictions5_t
	if EXTRATREE:
		
		ExtraTreeModel.fit(X_train,y_train)
		train_predictions6 = ExtraTreeModel.predict(X_train)
		dev_predictions6=ExtraTreeModel.predict(X_dev)
		# test_predictions6=ExtraTreeModel.predict(X_test)
		score = R_score(y_dev, dev_predictions6)
		print("ExtraTrees:\t{:.4f}".format(score))
		train_predictions6_t = train_predictions6.reshape((train_N,1))
		dev_predictions6_t = dev_predictions6.reshape((dev_N,1))
		# test_predictions6_t = test_predictions6.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions6_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions6_t], axis=1)
		else:
			x_train_predictions = train_predictions6_t
			x_dev_predictions = dev_predictions6_t
	if ADABOOST:
		
		AdaBoostModel.fit(X_train,y_train)
		train_predictions7 = AdaBoostModel.predict(X_train)
		dev_predictions7=AdaBoostModel.predict(X_dev)
		# test_predictions7=AdaBoostModel.predict(X_test)
		score = R_score(y_dev, dev_predictions7)
		print("Adaboost:\t{:.4f}".format(score))
		train_predictions7_t = train_predictions7.reshape((train_N,1))
		dev_predictions7_t = dev_predictions7.reshape((dev_N,1))
		# test_predictions7_t = test_predictions7.reshape((test_N,1))
		if x_train_predictions is not None:
			x_train_predictions = np.concatenate([x_train_predictions, train_predictions7_t], axis=1)
			x_dev_predictions = np.concatenate([x_dev_predictions, dev_predictions7_t], axis=1)
		else:
			x_train_predictions = train_predictions7_t
			x_dev_predictions = dev_predictions7_t
	if MIXING:
		# TestModel =  MLPRegressor(alpha=0.01, hidden_layer_sizes=(32), max_iter=300)
		# TestModel =	Ridge(fit_intercept=False)
		if MIXING:
			TestModel.fit(x_train_predictions, y_train)
			dev_predictions=TestModel.predict(x_dev_predictions)
			score = R_score(y_dev, dev_predictions)
			print("overall:\t{:.4f}".format(score))
	# x_train_predictions = np.concatenate([train_predictions0_t, train_predictions1_t, train_predictions2_t, train_predictions3_t, train_predictions4_t, train_predictions5_t, train_predictions6_t, train_predictions7_t],axis=1)
	# x_dev_predictions = np.concatenate([dev_predictions0_t, dev_predictions1_t, dev_predictions2_t, dev_predictions3_t, dev_predictions4_t, dev_predictions5_t, dev_predictions6_t, dev_predictions7_t],axis=1)
for iii in [0.01]:
	# TestModel =  MLPRegressor(alpha=0.01, hidden_layer_sizes=(32), max_iter=300)
	# TestModel =	Ridge(fit_intercept=False,alpha=iii)
	# if MIXING:
	# 	TestModel.fit(x_train_predictions, y_train)
	# 	dev_predictions=TestModel.predict(x_dev_predictions)
	# 	score = R_score(y_dev, dev_predictions)
	# 	print iii,
	# 	print("overall:\t{:.4f}".format(score))

	# testing
	if TESTING:
		y_train_predict = None
		x_test_predictions=None
		if DECISIONTREE:
			# decision tree
			DecisionTreeModel.fit(X_train_all, y_train_all)
			y_train0=np.expand_dims(DecisionTreeModel.predict(X_train_all), axis=1)
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train0], axis=1)
			else:
				y_train_predict = y_train0
			test_predictions0=DecisionTreeModel.predict(X_test)
			test_predictions0_t = test_predictions0.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions0_t], axis=1)
			else:
				x_test_predictions = test_predictions0_t
		if RANDOMFOREST:
			# random forest
			RandomForestModel.fit(X_train_all, y_train_all)
			y_train1=np.expand_dims(RandomForestModel.predict(X_train_all), axis=1)
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train1], axis=1)
			else:
				y_train_predict = y_train1
			test_predictions1=RandomForestModel.predict(X_test)
			test_predictions1_t = test_predictions1.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions1_t], axis=1)
			else:
				x_test_predictions = test_predictions1_t
		if XGBOOST:
			# xg boost
			XGBoostModel.fit(X_train_all, y_train_all)
			y_train2=np.expand_dims(XGBoostModel.predict(X_train_all), axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train2], axis=1)
			else:
				y_train_predict = y_train2
			test_predictions2=XGBoostModel.predict(X_test)
			test_predictions2_t = test_predictions2.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions2_t], axis=1)
			else:
				x_test_predictions = test_predictions2_t
		if GRADIENTBOOST:
			# gradient boost
			GradientBoostingModel.fit(X_train_all, y_train_all)
			y_train3=np.expand_dims(GradientBoostingModel.predict(X_train_all), axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train3], axis=1)
			else:
				y_train_predict = y_train3			
			test_predictions3=GradientBoostingModel.predict(X_test)
			test_predictions3_t = test_predictions3.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions3_t], axis=1)
			else:
				x_test_predictions = test_predictions3_t
		if SVM:
			# svm
			SVMModel.fit(X_train_all, y_train_all)
			y_train4=np.expand_dims(SVMModel.predict(X_train_all), axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train4], axis=1)
			else:
				y_train_predict = y_train4
			test_predictions4=SVMModel.predict(X_test)
			test_predictions4_t = test_predictions4.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions4_t], axis=1)
			else:
				x_test_predictions = test_predictions4_t
		if KNN:
			# knn
			KNNModel.fit(X_train_all, y_train_all)
			y_train5=np.expand_dims(KNNModel.predict(X_train_all), axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train5], axis=1)
			else:
				y_train_predict = y_train5
			test_predictions5=KNNModel.predict(X_test)
			test_predictions5_t = test_predictions5.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions5_t], axis=1)
			else:
				x_test_predictions = test_predictions5_t
		if EXTRATREE:	
			# extra tree
			ExtraTreeModel.fit(X_train_all, y_train_all)
			y_train6=np.expand_dims(ExtraTreeModel.predict(X_train_all),axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train6], axis=1)
			else:
				y_train_predict = y_train6
			test_predictions6=ExtraTreeModel.predict(X_test)
			test_predictions6_t = test_predictions6.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions6_t], axis=1)
			else:
				x_test_predictions = test_predictions6_t
		if ADABOOST:
			# ada boost
			AdaBoostModel.fit(X_train_all, y_train_all)
			y_train7=np.expand_dims(AdaBoostModel.predict(X_train_all),axis=1)	
			if y_train_predict is not None:
				y_train_predict = np.concatenate([y_train_predict, y_train7], axis=1)
			else:
				y_train_predict = y_train7
			test_predictions7=AdaBoostModel.predict(X_test)
			test_predictions7_t = test_predictions7.reshape((test_N,1))
			if x_test_predictions is not None:
				x_test_predictions = np.concatenate([x_test_predictions, test_predictions7_t], axis=1)
			else:
				x_test_predictions = test_predictions7_t
		if MIXING:
			assert x_test_predictions is not None
			TestModel.fit(y_train_predict, y_train_all)
			# x_test_predictions = np.concatenate([test_predictions1_t, test_predictions2_t, test_predictions3_t],axis=1)
			test_predictions = TestModel.predict(x_test_predictions)

		np.savetxt("ml_method_all.csv", np.around(test_predictions), delimiter=",",fmt='%d')
