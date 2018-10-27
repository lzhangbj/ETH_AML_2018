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
import matplotlib.pyplot as plt
import csv

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as KB

# matplotlib.use("Agg")
# plt.switch_backend('Agg')

logger = logging.getLogger("model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
	dropout 	= 0.         
	input_size  = 887                                                                            
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
# class Model:
# 	def __init__(self, config):
# 		self.config = config

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
	'''
		1.read data: train
		2.use 10-fold to derived it into 10 training and dev data
		3.define different model class into set M={}: difference is the hidden_size, learning rate(3), l2, and dropout 4 element
			hidden_sizes: 	30, 	50, 	100
			learning_rates:	0.01, 	0.001, 	0.0001
			l2: 			0.001, 	0.0001
			dropout: 		1,		0.8,	0.5	
			batch_size:		32
			epochs:			10
			for each model:
				for each fold:
					1.input different training and dev data
					2.train on training data and develop on dev data
					3.get dev data score 
				average dev data score
			choose the best model and retrain it on all data
			store the model in model_path
	'''
	config = Config(args)
	feature_num = args.feature_num
	train_data = np.load(args.data_train)
	X_test = np.load(args.data_test)
	dev_data = np.load(args.data_dev)

	data_means = np.load(args.norm_mean)
	data_maxs = np.load(args.norm_max)

	X_train = ((train_data[:, 1:]-data_means)/data_maxs)
	y_train = train_data[:, 0]
	X_dev = ((dev_data[:, 1:]-data_means)/data_maxs)
	y_dev = dev_data[:, 0]
	X_train_all = np.concatenate([X_train, X_dev], axis=0)
	y_train_all = np.concatenate([y_train, y_dev], axis=0)
	X_test = ((X_test-data_means)/data_maxs)
	if feature_num!=887:
		feature_order = np.load(args.feature_order)
		assert feature_num <= feature_order.size
		feature_order = feature_order[:feature_num]
		X_train = X_train[:, feature_order]
		X_dev = X_dev[:, feature_order]
		X_train_all = X_train_all[:,feature_order]
		X_test = X_test[:, feature_order]
	assert X_train.shape[1] == config.input_size, "x: {}, {}".format(X_train.shape[0], X_train.shape[1])

	model = Sequential()
	if config.cell == "One_layer_nn":
		# model.add(Dropout(config.dropout))
		model.add(Dense(config.hidden_size, input_dim=config.input_size, activation='relu', 
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
			kernel_regularizer=keras.regularizers.l2(config.l2),
			use_bias=True, bias_initializer=keras.initializers.Zeros(),
			))
		model.add(Dropout(config.dropout))
		model.add(Dense(1,  
			kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
			kernel_regularizer=keras.regularizers.l2(config.l2),
			use_bias=True, bias_initializer=keras.initializers.Zeros(),
			))
		# model.add(Dropout(config.dropout))
		# model.add(Dense(1,  
		# 	kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
		# 	kernel_regularizer=keras.regularizers.l2(config.l2),
		# 	use_bias=True, bias_initializer=keras.initializers.Zeros(),
		# 	))			
	else:
		raise NameError('CELL NAME ERROR')
	# Compile model
	# opt = keras.optimizers.Adam(lr=config.lr)
	opt = keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	def R_score(y_true, y_pred):
	    SS_res =  KB.sum(KB.square( y_true-y_pred ))
	    SS_tot = KB.sum(KB.square( y_true - KB.mean(y_true) ) )
	    return ( 1 - SS_res/(SS_tot + KB.epsilon()) )
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy", R_score])#"accuracy"

	# Fit the model
	best_score = 0.
	for i in range(config.epochs):
		print "EPOCH "+str(i),
		model.fit(X_train, y_train, epochs=1, batch_size=config.batch_size, verbose=0)
		# calculate predictions
		R_score = model.evaluate(X_dev, y_dev,verbose=0)[2]
		print("eval score: {:.4f}\tbest score: {:.4f}".format(R_score,best_score))
		if R_score > best_score:
			best_score=R_score
			model.save("best_model.h5")
	# round predictions
	# model.fit(X_train_all, y_train_all, epochs=150, batch_size=config.batch_size, verbose=2)

	result = model.predict(X_test)
	np.savetxt("foo.csv", result, delimiter=",",fmt='%.4f')
	 
def test(args):
	pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Trains and tests model')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('train', help='')
	command_parser.add_argument('-dt', '--data-train', type=str, default="data/train.npy", help="Training data")
	command_parser.add_argument('-dd', '--data-dev', type=str, default="data/dev.npy", help="Training data")
	command_parser.add_argument('-dtt', '--data-test', type=str, default="data/test.npy", help="Training data")
	command_parser.add_argument('-fo', '--feature-order', type=str, default="data/feature_order.npy", help="Training data")
	command_parser.add_argument('-fon', '--feature-num', type=int, default=887, help="Training data")
	command_parser.add_argument('-mean', '--norm-mean', type=str, default="data/train_means.npy", help="Training data")
	command_parser.add_argument('-max', '--norm-max', type=str, default="data/train_maxs.npy", help="Training data")
	command_parser.add_argument('-lr', '--lr', type=float, 		default=0.001, help="Learning rate")
	command_parser.add_argument('-l2','--l2', type=float, 		default = 0.0001,help="lamda value") 
	command_parser.add_argument('-c','--cell', type=str, choices=["One_layer_nn"],	default ="One_layer_nn", help="cell choices") 
	command_parser.add_argument('-e', '--epochs', type=int, 	default=10, help="epoch")
	command_parser.add_argument('-b','--batch_size', type=int, 	default=30,help="batch size value") 
	command_parser.add_argument('-dr', '--dropout', type=float, default=0, help="drop out")
	command_parser.add_argument('-gpu','--gpu', choices=['','0','1','2','3'], default='', help="specify gpu use")  
	command_parser.set_defaults(func=train)

	command_parser = subparsers.add_parser('test', help='')
	command_parser.add_argument('-dt', '--data-test', type=str, default="data/test.csv", help="Training data")
	command_parser.add_argument('-mp', '--model-path', type=str,  help="model path")
	command_parser.add_argument('-gpu','--gpu', choices=['','0','1','2','3'], default='', help="specify gpu use")  
	command_parser.set_defaults(func=test)


	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)
