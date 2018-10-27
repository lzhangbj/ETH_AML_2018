import tensorflow as tf
import argparse
import logging
import sys
import pandas as pd
import time
from datetime import datetime
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import backend as KB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from tensorflow.python.client import device_lib

config = tf.ConfigProto(device_count = {'GPU': 2} ) 
sess = tf.Session(config=config) 
KB.set_session(sess)

print(device_lib.list_local_devices())
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

# import keras.losses 


# matplotlib.use("Agg")
# plt.switch_backend('Agg')

logger = logging.getLogger("model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def My_score(y_true, y_pred):
	y_true_ = tf.argmax(y_true, axis=1)
	y_pred_ = tf.argmax(y_pred, axis=1)
	# y = KB.concatenate([y_true_[:, tf.newaxis], y_pred_[:, tf.newaxis]], axis=1)

	# f_case = KB.gather(y,tf.where(y[:,0]==0))
	# s_case = KB.gather(y,tf.where(y[:,0]==1))
	# t_case = KB.gather(y,tf.where(y[:,0]==2))
	my_recall = [0] * 3

	def rrecall(y_true, y_pred):	
	    """Recall metric.	
	     Only computes a batch-wise average of recall.	
	     Computes the recall, a metric for multi-label classification of	
	    how many relevant items are selected.	
	    """	
	    true_positives = KB.sum(KB.round(KB.clip(y_true * y_pred, 0, 1)))	
	    possible_positives = KB.sum(KB.round(KB.clip(y_true, 0, 1)))	
	    recall = true_positives / (possible_positives + KB.epsilon())	
	    return recall

	for k in range(3):
		y1 = KB.equal(y_true_, k)
		y1 = KB.cast(y1,'float32')		
		y2 = KB.equal(y_pred_, k)
		y2 = KB.cast(y2,'float32')
		my_recall[k]= rrecall(y1,y2)
	value = (my_recall[0]+my_recall[1]+my_recall[2])/3
	return value

class Config:
	dropout 	= 0.         
	input_size  = 1000                                                                            
	hidden_size = 350                                                                                     
	batch_size = 30                                                                                           
	epochs = 10                                                                                                                                                                          #hyper
	lr = 0.001 
	l2 = 0.0001                                                                                         
	gpu = ''  

	def __init__(self, args):
		self.cell = args.cell
		self.gpu = args.gpu
		self.input_size=args.feature_num
		try:
			self.lr = args.lr
			self.l2 = args.l2
			self.batch_size = args.batch_size
			self.dropout = args.dropout
			self.epochs = args.epochs
		except:
			pass
		if "model_path" in args:
			self.output_path = args.model_path
		else:
			self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
		self.model_output = self.output_path + "model.weights"
		self.eval_output = self.output_path + "results.txt"
		self.log_output = self.output_path + "log.txt"

def load_csv_data(file_dir):
	data = pd.read_csv(file_dir, sep=',',header=None, dtype=float,skiprows=1).values[:,1:]
	print("loading {} {} data from {}".format(data.shape[0], data.shape[1], file_dir))
	return data

def cross_validate(data, folds):
	num_data = data.size[0]
	assert num_data>=folds
	fold_content = num_data/fold
	data_shuffled = np.shuffle(data)
	data_folds = []
	for i in range(folds):
		data_folds.append(data_shuffled[i*fold_content:(i+1)*fold_content])
	assert data_folds.size[0] == folds
	return data_folds



def train(args):
	scaler = StandardScaler()
	config = Config(args)
	feature_num = args.feature_num
	train_data = np.load(args.data_train)
	X_test = np.load(args.data_test)
	dev_data = np.load(args.data_dev)

	train_data_all = np.concatenate([train_data, dev_data], axis=0)
	np.random.shuffle(train_data_all)
	train_data_folds = np.array_split(train_data_all, 10, axis=0)

	X_train_folds = []
	X_dev_folds = []
	X_test_folds = []
	y_train_folds = []
	y_dev_folds = []
	y_test_folds = []
	scalers = []

	for fold in range(10):
		np.random.shuffle(train_data_folds[fold])
		test_fold = train_data_folds[fold]
		dev_fold = train_data_folds[(fold+1)%10]
		train_fold = [train_data_folds[i] for i in range(10) if i != fold and i != (fold+1)%10]
		train_fold = np.concatenate(train_fold, axis=0)

		X_train_fold = train_fold[:, 1:]
		y_train_fold = train_fold[:, 0].astype(int)
		X_dev_fold = dev_fold[:, 1:]
		y_dev_fold = dev_fold[:, 0].astype(int)
		X_test_fold = test_fold[:, 1:]
		y_test_fold = test_fold[:, 0].astype(int)

		scaler = StandardScaler()
		X_train_fold = scaler.fit_transform(X_train_fold)
		X_dev_fold   = scaler.transform(X_dev_fold)
		X_test_fold  = scaler.transform(X_test_fold)

		scalers.append(scaler)

		y_train_fold_ = np.zeros((len(y_train_fold), 3))
		y_train_fold_[range(len(y_train_fold)),y_train_fold]=1
		y_train_fold = y_train_fold_
		y_dev_fold_ = np.zeros((len(y_dev_fold), 3))
		y_dev_fold_[range(len(y_dev_fold)),y_dev_fold]=1
		y_dev_fold = y_dev_fold_
		y_test_fold_ = np.zeros((len(y_test_fold), 3))
		y_test_fold_[range(len(y_test_fold)),y_test_fold]=1
		y_test_fold = y_test_fold_

		X_train_folds.append(X_train_fold)
		y_train_folds.append(y_train_fold)
		X_dev_folds.append(X_dev_fold)
		y_dev_folds.append(y_dev_fold)
		X_test_folds.append(X_test_fold)
		y_test_folds.append(y_test_fold)

	number=1
	dropout_1_range = \
		[[0,	0,	0],\
		[0,	0.1,	0],\
		[0,	0.1,	0.1],\
		[0,	0.2,	0],\
		[0,	0.2,	0.1],\
		[0,	0.3,	0],\
		[0,	0.3,	0.1],\
		[0,	0.3,	0.2],\
		[0,	0.5,	0],\
		[0,	0.5,	0.1],\
		[0,	0.5,	0.3],\
		[0,	0.5,	0.5],\
		[0,	0.7,	0.1],\
		[0,	0.7,	0.3],\
		[0,	0.7,	0.5]]
	dropout_2_range = [\
		[0.1,	0,	0],\
		[0.1,	0.1,	0],\
		[0.1,	0.1,	0.1],\
		[0.1,	0.2,	0],\
		[0.1,	0.2,	0.1],\
		[0.1,	0.2,	0.2],\
		[0.1,	0.3,	0],\
		[0.1,	0.3,	0.1],\
		[0.1,	0.3,	0.2],\
		[0.1,	0.5,	0],\
		[0.1,	0.5,	0.1],\
		[0.1,	0.5,	0.3],\
		[0.1,	0.5,	0.4],\
		[0.1,	0.7,	0.1],\
		[0.1,	0.7,	0.3],\
		[0.1,	0.7,	0.5]]
	dropout_3_range = [\
		[0.2,	0,	0],\
		[0.2,	0.1,	0],\
		[0.2,	0.1,	0.1],\
		[0.2,	0.2,	0],\
		[0.2,	0.2,	0.1],\
		[0.2,	0.2,	0.2],\
		[0.2,	0.3,	0],\
		[0.2,	0.3,	0.1],\
		[0.2,	0.3,	0.2],\
		[0.2,	0.5,	0],\
		[0.2,	0.5,	0.1],\
		[0.2,	0.5,	0.3],\
		[0.2,	0.5,	0.4],\
		[0.2,	0.7,	0.1],\
		[0.2,	0.7,	0.3],\
		[0.2,	0.7,	0.5]]
	dropout_4_range = [\
		[0.3,	0,	0],\
		[0.3,	0.1,	0],\
		[0.3,	0.1,	0.1],\
		[0.3,	0.2,	0],\
		[0.3,	0.2,	0.1],\
		[0.3,	0.2,	0.2],\
		[0.3,	0.3,	0],\
		[0.3,	0.3,	0.1],\
		[0.3,	0.3,	0.2],\
		[0.3,	0.5,	0],\
		[0.3,	0.5,	0.1],\
		[0.3,	0.5,	0.3],\
		[0.3,	0.5,	0.4],\
		[0.3,	0.7,	0.1],\
		[0.3,	0.7,	0.3],\
		[0.3,	0.7,	0.5]]
	if number==1:
		dropout_range = dropout_1_range
	elif number==2:
		dropout_range = dropout_2_range
	elif number==3:
		dropout_range = dropout_3_range
	elif number==4:
		dropout_range = dropout_4_range

	for dropout1, dropout2, dropout3 in dropout_range:
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		print(">>>>>>>>>>>>>>> {} {} {} <<<<<<<<<<<<<<<<<<<<<<<".format(dropout1, dropout2, dropout3))
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		model = Sequential()
		model.add(Dropout(dropout1,input_shape=(config.input_size,)))
		model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
			kernel_regularizer=keras.regularizers.l2(config.l2),
			use_bias=True, bias_initializer=keras.initializers.Zeros(),
			))
		model.add(Dropout(dropout2))
		model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
			kernel_regularizer=keras.regularizers.l2(config.l2),
			use_bias=True, bias_initializer=keras.initializers.Zeros(),
			))
		model.add(Dropout(dropout3))	
		model.add(Dense(3,  activation='softmax',
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
			kernel_regularizer=keras.regularizers.l2(config.l2),
			use_bias=True, bias_initializer=keras.initializers.Zeros(),
			))		
		# Compile model
		opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
		model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[My_score])#"accuracy"
		init_weights = model.get_weights()
		# Fit the model
		prediction_tot = None
		test_predictions = None
		glob_score = 0
		for k in range(10):
			best_score = 0.
			model.set_weights(init_weights)
			for i in range(50):
				model.fit(X_train_folds[k], y_train_folds[k], epochs=1, batch_size=config.batch_size, verbose=0)
				R_score = model.evaluate(X_dev_folds[k], y_dev_folds[k],verbose=0)[1]
				if R_score > best_score:
					test_prediction = model.predict(X_test_folds[k])
					best_score=R_score
				print("\r<<<<<<< loading {} / 50	score:{:.4f} best:{:.4f}>>>>>>>".format(i+1, R_score, best_score),end='') 
				sys.stdout.flush()			
			if test_predictions is not None:
				test_predictions = np.concatenate([test_predictions,test_prediction], axis=0)
			else:
				test_predictions = test_prediction
			test_score = balanced_accuracy_score(np.argmax(y_test_folds[k],axis=1), np.argmax(test_prediction,axis=1))
			print("\ttest score: {:.4f}".format(test_score))
		glob_score = balanced_accuracy_score(np.argmax(np.concatenate(y_test_folds, axis=0), axis=1), np.argmax(test_predictions,axis=1))
		print("<<<<<<< {} {} {} final score: {} >>>>>>>>".format(dropout1, dropout2, dropout3, glob_score))
		del model
		print()


def get_cv_folds(data_all, data_test):
	'''
		return 5 values:
		 X_train_folds, y_train_folds, X_dev_folds, y_dev_folds, test_folds(scaled test data) 
	'''	
	np.random.shuffle(data_all)
	train_data_folds = np.array_split(data_all, 10, axis=0)

	X_train_folds = []
	X_dev_folds = []
	y_train_folds = []
	y_dev_folds = []
	test_folds = []

	for fold in range(10):
		np.random.shuffle(train_data_folds[fold])
		dev_fold = train_data_folds[fold]
		train_fold = [train_data_folds[i] for i in range(10) if i != fold ]
		train_fold = np.concatenate(train_fold, axis=0)

		X_train_fold = train_fold[:, 1:]
		y_train_fold = train_fold[:, 0].astype(int)
		X_dev_fold = dev_fold[:, 1:]
		y_dev_fold = dev_fold[:, 0].astype(int)

		scaler = StandardScaler()
		X_train_fold = scaler.fit_transform(X_train_fold)
		X_dev_fold   = scaler.transform(X_dev_fold)
		X_test_fold  = scaler.transform(data_test)

		y_train_fold_ = np.zeros((len(y_train_fold), 3))
		y_train_fold_[range(len(y_train_fold)),y_train_fold]=1
		y_train_fold = y_train_fold_
		y_dev_fold_ = np.zeros((len(y_dev_fold), 3))
		y_dev_fold_[range(len(y_dev_fold)),y_dev_fold]=1
		y_dev_fold = y_dev_fold_

		X_train_folds.append(X_train_fold)
		y_train_folds.append(y_train_fold)
		X_dev_folds.append(X_dev_fold)
		y_dev_folds.append(y_dev_fold)
		test_folds.append(X_test_fold)

	return X_train_folds, y_train_folds, X_dev_folds, y_dev_folds, test_folds

def get_boostrap_fold(data_all, data_test):
	'''
		return 5 values:
		 X_train, y_train, X_dev, y_dev, test(scaled test data) 			
	'''
	size = len(data_all)
	train_size = int(size*9/10)
	dev_size   = int(size/10)

	np.random.shuffle(data_all)
	index_choice = range(size)
	dev_indices = np.random.choice(index_choice, size=dev_size, replace=True)
	train_indices = [i for i in index_choice if i not in dev_indices] 
	train_data = data_all[train_indices]
	dev_data = data_all[dev_indices]

	X_train = train_data[:, 1:]
	y_train = train_data[:, 0].astype(int)
	X_dev = dev_data[:, 1:]
	y_dev = dev_data[:, 0].astype(int)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_dev   = scaler.transform(X_dev)
	X_test  = scaler.transform(data_test)

	y_train_ = np.zeros((len(y_train), 3))
	y_train_[range(len(y_train)),y_train]=1
	y_train = y_train_
	y_dev_ = np.zeros((len(y_dev), 3))
	y_dev_[range(len(y_dev)),y_dev]=1
	y_dev = y_dev_

	return X_train, y_train, X_dev, y_dev, X_test

def test(args):
	'''
		we compare two methods: bootstrap and multiple cross validation
	'''
	scaler = StandardScaler()
	config = Config(args)
	train_data = np.load(args.data_train)
	X_test = np.load(args.data_test)
	dev_data = np.load(args.data_dev)

	train_data_all = np.concatenate([train_data, dev_data], axis=0)

	dropout1 = args.dropout1
	dropout2 = args.dropout2
	dropout3 = args.dropout3

	model = Sequential()
	model.add(Dropout(dropout1,input_shape=(config.input_size,)))
	model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))
	model.add(Dropout(dropout2))
	model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))
	model.add(Dropout(dropout3))	
	model.add(Dense(3,  activation='softmax',
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))		
	# Compile model
	opt = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[My_score])#"accuracy"
	init_weights = model.get_weights()
	# Fit the model
	test_prediction_tot = None
	test_prediction_average = 0
	dev_prediction_tot = None
	dev_label_tot = None
	glob_best_score = 0
	class_weights={0:6.,1:1.,2:6.}
	if args.method == "bootstrap":
		for iter in range(args.iteration): 
			best_score = 0.
			model.set_weights(init_weights)
			X_train, y_train, X_dev, y_dev, data_test = get_boostrap_fold(train_data_all, X_test)
			for i in range(100):
				model.fit(X_train, y_train, epochs=1, batch_size=config.batch_size, verbose=0,class_weight=class_weights)
				R_score = model.evaluate(X_dev, y_dev,verbose=0)[1]
				if R_score > best_score:
					test_prediction_ = model.predict(data_test)
					test_prediction = np.argmax(test_prediction_, axis=1)[:,np.newaxis]
					dev_prediction = model.predict(X_dev)
					# model.save("boot_strap_{}_{}_{}_iter_{}.h5".format(dropout1, dropout2, dropout3, iter))
					best_score=R_score
				print("\r<<<<<<< loading {} / 100	score:{:.4f} best:{:.4f}>>>>>>>".format(i+1, R_score, best_score), end='')
				sys.stdout.flush()
			if best_score>glob_best_score:
				glob_best_score=best_score
				model.save("boot_strap_{}_{}_{}_lr_{}_iter_{}_best.h5".format(dropout1, dropout2, dropout3, args.lr, args.iteration))
			test_prediction_average += test_prediction_

			if test_prediction_tot is not None:
				test_prediction_tot = np.concatenate([test_prediction_tot,test_prediction], axis=1)
			else:
				test_prediction_tot = test_prediction

			if dev_prediction_tot is not None:
				dev_prediction_tot = np.concatenate([dev_prediction_tot,dev_prediction], axis=0)
			else:
				dev_prediction_tot = dev_prediction

			if dev_label_tot is not None:
				dev_label_tot = np.concatenate([dev_label_tot,y_dev], axis=0)
			else:
				dev_label_tot = y_dev
			print()
			dev_score = balanced_accuracy_score(np.argmax(dev_label_tot,axis=1), np.argmax(dev_prediction_tot,axis=1))
			print("<<<<<<< {} {} {} lr {:.6f} iteration {} cumulated score: {} >>>>>>>>".format(dropout1, dropout2, dropout3, args.lr, iter, dev_score))
			print()
			sys.stdout.flush()	
		test_prediction_tot = test_prediction_tot.astype(int)
		print(test_prediction_tot.shape)
		test_prediction_tot = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 1, test_prediction_tot)
		test_prediction_tot = np.argmax(test_prediction_tot,axis=1)
		test_prediction_tot = test_prediction_tot.astype(int)
		test_prediction_average/=args.iteration
		test_prediction_average=np.argmax(test_prediction_average,axis=1)
		#save as vote
		d = {"id": range(4100), "y": test_prediction_tot}
		keys = sorted(d.keys())
		with open("bootstrap_vote_{}_{}_{}_lr_{:.6f}_iter_{}_foo.csv".format(dropout1, dropout2, dropout3, args.lr, args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		#save as average
		d = {"id": range(4100), "y": test_prediction_average}
		keys = sorted(d.keys())
		with open("bootstrap_avr_{}_{}_{}_lr_{:.6f}_iter_{}_foo.csv".format(dropout1, dropout2, dropout3, args.lr, args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		sys.stdout.flush()	


	elif args.method == "cv":
		for iter in range(args.iteration):
			X_train_folds, y_train_folds, X_dev_folds, y_dev_folds, test_folds = get_cv_folds(train_data_all, X_test)
			for k in range(10):
				best_score = 0.
				model.set_weights(init_weights)
				for i in range(100):
					model.fit(X_train_folds[k], y_train_folds[k], epochs=1, batch_size=config.batch_size, verbose=0,class_weight=class_weights)
					R_score = model.evaluate(X_dev_folds[k], y_dev_folds[k],verbose=0)[1]
					if R_score > best_score:
						test_prediction_ = model.predict(test_folds[k])
						test_prediction = np.argmax(test_prediction_, axis=1)[:,np.newaxis]
						dev_prediction = model.predict(X_dev_folds[k])
						# model.save("cv_{}_{}_{}_iter_{}_k_{}.h5".format(dropout1, dropout2, dropout3, iter, k))
						best_score=R_score
					print("\r<<<<<<< loading {} / 100	score:{:.4f} best:{:.4f}>>>>>>>".format(i+1, R_score, best_score), end='')
					sys.stdout.flush()
				if best_score>glob_best_score:
					glob_best_score=best_score
					model.save("cv_{}_{}_{}_lr_{}_iter_{}_best.h5".format(dropout1, dropout2, dropout3, args.lr, args.iteration))
				test_prediction_average += test_prediction_

				if test_prediction_tot is not None:
					test_prediction_tot = np.concatenate([test_prediction_tot,test_prediction], axis=1)
				else:
					test_prediction_tot = test_prediction

				if dev_prediction_tot is not None:
					dev_prediction_tot = np.concatenate([dev_prediction_tot,dev_prediction], axis=0)
				else:
					dev_prediction_tot = dev_prediction

				if dev_label_tot is not None:
					dev_label_tot = np.concatenate([dev_label_tot,y_dev_folds[k]], axis=0)
				else:
					dev_label_tot = y_dev_folds[k]
				print()
				dev_score = balanced_accuracy_score(np.argmax(dev_label_tot,axis=1), np.argmax(dev_prediction_tot,axis=1))
				print("<<<<<<< {} {} {} lr {:.6f} iteration {} cumulated score: {} >>>>>>>>".format(dropout1, dropout2, dropout3, args.lr, iter, dev_score))
				print()
				sys.stdout.flush()	
		test_prediction_tot = test_prediction_tot.astype(int)
		test_prediction_tot = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 1, test_prediction_tot)
		test_prediction_tot = np.argmax(test_prediction_tot,axis=1)
		test_prediction_tot = test_prediction_tot.astype(int)
		test_prediction_average/=(args.iteration*10)
		test_prediction_average=np.argmax(test_prediction_average,axis=1)
		d = {"id": range(4100), "y": test_prediction_tot}
		keys = sorted(d.keys())
		with open("cv_vote_{}_{}_{}_lr_{:.6f}_iter_{}foo.csv".format(dropout1, dropout2, dropout3,args.lr, args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		#save as average
		d = {"id": range(4100), "y": test_prediction_average}
		keys = sorted(d.keys())
		with open("cv_avr_{}_{}_{}_lr_{:.6f}_iter_{}_foo.csv".format(dropout1, dropout2, dropout3, args.lr, args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		sys.stdout.flush()		

def apply_meam(args):
	'''
		we compare two methods: bootstrap and multiple cross validation
	'''
	scaler = StandardScaler()
	config = Config(args)
	train_data = np.load(args.data_train)
	X_test = np.load(args.data_test)
	dev_data = np.load(args.data_dev)

	train_data_all = np.concatenate([train_data, dev_data], axis=0)

	dropout1 = args.dropout1
	dropout2 = args.dropout2
	dropout3 = args.dropout3

	model = Sequential()
	model.add(Dropout(dropout1,input_shape=(config.input_size,)))
	model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))
	model.add(Dropout(dropout2))
	model.add(Dense(1024, input_dim=config.input_size, activation='relu', 
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))
	model.add(Dropout(dropout3))	
	model.add(Dense(3,  activation='softmax',
		kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		kernel_regularizer=keras.regularizers.l2(config.l2),
		use_bias=True, bias_initializer=keras.initializers.Zeros(),
		))		
	# Compile model
	opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[My_score])#"accuracy"
	init_weights = model.get_weights()
	# Fit the model
	test_prediction_tot = None
	dev_prediction_tot = None
	dev_label_tot = None
	glob_score = 0
	if args.method == "bootstrap":
		for iter in range(args.iteration): 
			best_score = 0.
			model.set_weights(init_weights)
			X_train, y_train, X_dev, y_dev, data_test = get_boostrap_fold(train_data_all, X_test)
			for i in range(50):
				model.fit(X_train, y_train, epochs=1, batch_size=config.batch_size, verbose=0)
				R_score = model.evaluate(X_dev, y_dev,verbose=0)[1]
				if R_score > best_score:
					test_prediction = model.predict(data_test)
					test_prediction = np.argmax(test_prediction, axis=1)[:,np.newaxis]
					dev_prediction = model.predict(X_dev)
					model.save("boot_strap_{}_{}_{}_iter_{}.h5".format(dropout1, dropout2, dropout3, iter))
					best_score=R_score
				print("\r<<<<<<< loading {} / 50	score:{:.4f} best:{:.4f}>>>>>>>".format(i+1, R_score, best_score), end='')
				sys.stdout.flush()

			if test_prediction_tot is not None:
				test_prediction_tot = np.concatenate([test_prediction_tot,test_prediction], axis=1)
			else:
				test_prediction_tot = test_prediction

			if dev_prediction_tot is not None:
				dev_prediction_tot = np.concatenate([dev_prediction_tot,dev_prediction], axis=0)
			else:
				dev_prediction_tot = dev_prediction

			if dev_label_tot is not None:
				dev_label_tot = np.concatenate([dev_label_tot,y_dev], axis=0)
			else:
				dev_label_tot = y_dev
			print()
			dev_score = balanced_accuracy_score(np.argmax(dev_label_tot,axis=1), np.argmax(dev_prediction_tot,axis=1))
			print("<<<<<<< {} {} {} iteration {} cumulated score: {} >>>>>>>>".format(dropout1, dropout2, dropout3, iter, dev_score))
			print()
			sys.stdout.flush()	
		test_prediction_tot = test_prediction_tot.astype(int)
		print(test_prediction_tot.shape)
		test_prediction_tot = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 1, test_prediction_tot)
		test_prediction_tot = np.argmax(test_prediction_tot,axis=1)
		test_prediction_tot = test_prediction_tot.astype(int)
		d = {"id": range(4100), "y": test_prediction_tot}
		keys = sorted(d.keys())
		with open("bootstrap_vote_{}_{}_{}_iter_{}_foo.csv".format(dropout1, dropout2, dropout3, args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		sys.stdout.flush()		

	elif args.method == "cv":
		for iter in range(args.iteration):
			X_train_folds, y_train_folds, X_dev_folds, y_dev_folds, test_folds = get_cv_folds(train_data_all, X_test)
			for k in range(10):
				best_score = 0.
				model.set_weights(init_weights)
				for i in range(50):
					model.fit(X_train_folds[k], y_train_folds[k], epochs=1, batch_size=config.batch_size, verbose=0)
					R_score = model.evaluate(X_dev_folds[k], y_dev_folds[k],verbose=0)[1]
					if R_score > best_score:
						test_prediction = model.predict(test_folds[k])
						test_prediction = np.argmax(test_prediction, axis=1)[:,np.newaxis]
						dev_prediction = model.predict(X_dev_folds[k])
						model.save("cv_{}_{}_{}_iter_{}_k_{}.h5".format(dropout1, dropout2, dropout3, iter, k))
						best_score=R_score
					print("\r<<<<<<< loading {} / 50	score:{:.4f} best:{:.4f}>>>>>>>".format(i+1, R_score, best_score), end='')
					sys.stdout.flush()

				if test_prediction_tot is not None:
					test_prediction_tot = np.concatenate([test_prediction_tot,test_prediction], axis=1)
				else:
					test_prediction_tot = test_prediction
		test_prediction_tot = test_prediction_tot.astype(int)
		test_prediction_tot = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 1, test_prediction_tot)
		test_prediction_tot = np.argmax(test_prediction_tot,axis=1)
		test_prediction_tot = test_prediction_tot.astype(int)
		d = {"id": range(4100), "y": test_prediction_tot}
		keys = sorted(d.keys())
		with open("cv_vote_{}_{}_{}_iter_{}foo.csv".format(dropout1, dropout2, dropout3,args.iteration), "w") as outfile:
			writer = csv.writer(outfile, delimiter = ",")
			writer.writerow(keys)
			writer.writerows(zip(*[d[key] for key in keys]))
		sys.stdout.flush()			


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Trains and tests model')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('train', help='')
	command_parser.add_argument('-dt', '--data-train', type=str, default="data/train.npy", help="Training data")
	command_parser.add_argument('-dd', '--data-dev', type=str, default="data/dev.npy", help="Training data")
	command_parser.add_argument('-dtt', '--data-test', type=str, default="data/test.npy", help="Training data")
	command_parser.add_argument('-fo', '--feature-order', type=str, default=None, help="Training data")
	command_parser.add_argument('-fon', '--feature-num', type=int, default=1000, help="Training data")
	command_parser.add_argument('-lr', '--lr', type=float, 		default=0.001, help="Learning rate")
	command_parser.add_argument('-l2','--l2', type=float, 		default = 0.01,help="lamda value") 
	command_parser.add_argument('-c','--cell', type=str, choices=["One_layer_nn"],	default ="One_layer_nn", help="cell choices") 
	command_parser.add_argument('-e', '--epochs', type=int, 	default=10, help="epoch")
	command_parser.add_argument('-b','--batch_size', type=int, 	default=30,help="batch size value") 
	command_parser.add_argument('-dr', '--dropout', type=float, default=0, help="drop out")
	command_parser.add_argument('-gpu','--gpu', choices=['','0','1','2','3'], default='', help="specify gpu use")  
	command_parser.set_defaults(func=train)

	command_parser = subparsers.add_parser('test', help='')
	command_parser.add_argument('-dt', '--data-train', type=str, default="data/train.npy", help="Training data")
	command_parser.add_argument('-dd', '--data-dev', type=str, default="data/dev.npy", help="Training data")
	command_parser.add_argument('-dtt', '--data-test', type=str, default="data/test.npy", help="Training data")
	command_parser.add_argument('-fo', '--feature-order', type=str, default=None, help="Training data")
	command_parser.add_argument('-fon', '--feature-num', type=int, default=1000, help="Training data")
	command_parser.add_argument('-lr', '--lr', type=float, 		default=0.0001, help="Learning rate")
	command_parser.add_argument('-l2','--l2', type=float, 		default = 0.01,help="lamda value") 
	command_parser.add_argument('-c','--cell', type=str, choices=["One_layer_nn"],	default ="One_layer_nn", help="cell choices") 
	command_parser.add_argument('-e', '--epochs', type=int, 	default=10, help="epoch")
	command_parser.add_argument('-b','--batch_size', type=int, 	default=30,help="batch size value") 
	command_parser.add_argument('-dr', '--dropout', type=float, default=0, help="drop out")
	command_parser.add_argument('-gpu','--gpu', choices=['','0','1','2','3'], default='', help="specify gpu use")  
	command_parser.add_argument('-m','--method', choices=["cv", "bootstrap"], default='bootstrap', help="specify gpu use")  
	command_parser.add_argument('-i','--iteration', type=int, default=50, help="specify gpu use")  
	command_parser.add_argument('-dr1','--dropout1', type=float, default=0, help="specify gpu use")  
	command_parser.add_argument('-dr2','--dropout2', type=float, default=0, help="specify gpu use")  
	command_parser.add_argument('-dr3','--dropout3', type=float, default=0, help="specify gpu use")  
	command_parser.set_defaults(func=test)


	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)
