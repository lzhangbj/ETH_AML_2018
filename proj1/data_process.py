import numpy as np
import pandas as pd
import argparse

def load_data(file_dir):
	x_train = pd.read_csv(file_dir+"/X_train.csv", sep=',',header=None, dtype=float, usecols=range(1,888),skiprows=1).values
	y_train = pd.read_csv(file_dir+"/y_train.csv", sep=',',header=None, dtype=float, usecols=(1,), skiprows=1).values#fillna(value = 0.).values
	x_test = pd.read_csv(file_dir+"/X_test.csv", sep=',',header=None, dtype=float, usecols=range(1,888),skiprows=1).values #fillna(value = 0.).values
	print(x_train)
	print(x_train.shape)
	print(y_train.shape)
	data = np.concatenate((y_train, x_train), axis=1)
	return data, x_test

def get_train_dev(data):
	length = data.shape[0]
	dev_length = length/10
	train_length = length - dev_length
	np.random.shuffle(data)
	dev = data[0:dev_length]
	train = data[dev_length:]
	return train, dev

def get_normalize_param(data):
	data_ = data[:, 1:]
	length = data.shape[0]
	means = np.mean(data_, axis=0)
	maxs = np.max(np.absolute(data_), axis=0)
	maxs[maxs == 0] = 1 
	return means, maxs

def read_in_features(file):
	order_data = pd.read_csv(file, sep=',',header=None, dtype=int, usecols=(0,),skiprows=1).values
	print(order_data.shape)
	length = order_data.shape[0]
	order_data = order_data.reshape((length))
	print(order_data.shape)
	np.save("data/feature_order.npy", order_data)
	return order_data


data, test_data = load_data("data")
train, dev = get_train_dev(data)
# print(train.shape)
# print(dev.shape)
np.save("data/train.npy", train)
np.save("data/dev.npy", dev)
np.save("data/test.npy",test_data)

# read_in_features("featuresSelectionWRTPearson.csv")