import tensorflow as tf
import numpy as np
import csv


'''
My notes:

Neural networks don't work well for this task. I think these might be reasons:
1. Neural networks are intrinsically not so good at regression tasks compared to classification tasks.
2. In this task, values in different fields range extremely large. Even when initial parameters are very
   small, loss can still expode before the networks can learn something. Pre-processing helps, but not much
2. Too little data: ~1K in total, only slight higher than data dimension. Neural networks are powerful and
   can efficiently and effectively reduce data dimensionality only when large amount of data is fed,
   otherwise in this task the irrelevant fields bring too much noise

Some possible improvements (if we insist on using neural networks)
1. Use PCA to extract features. Feed these feature to the networks
2. Use more data (number of samples must me significantly greater than data dimension)
'''


class BatchGenerator(object):

    def __init__(self, xs, ys, batch_size):
        self._xs = xs
        self._ys = ys
        self._batch_size = batch_size
        self._current_idx = 0
        self._num_batches = len(xs)//batch_size
        self._in_dim = len(xs[0])
        self._out_dim = len(ys[0])
        
        
    def num_batches(self):
        return self._num_batches


    def generate(self):
        xs = np.zeros((self._batch_size, self._in_dim), dtype=np.float32)
        ys = np.zeros((self._batch_size, self._out_dim), dtype=np.float32)
        while True:
            for i in range(self._batch_size):
                if self._current_idx == len(self._xs):
                    self._current_idx = 0

                xs[i] = self._xs[self._current_idx]
                ys[i] = self._ys[self._current_idx]
                self._current_idx += 1
            yield xs, ys



X_dict = {}
keys = []
with open('data/X_train.csv') as csvfile:
    x_train_reader = csv.DictReader(csvfile)
    fields = x_train_reader.fieldnames[1:]
    key_field = x_train_reader.fieldnames[0]
    for row in x_train_reader:
        key = row[key_field]
        keys.append(key)
        X_dict[key] = [float(row[field]) if row[field] else None for field in fields]

 
Y_dict = {}     
with open('data/y_train.csv') as csvfile:
    y_train_reader = csv.DictReader(csvfile)
    field = y_train_reader.fieldnames[-1]
    key_field = x_train_reader.fieldnames[0]
    for row in y_train_reader:
        key = row[key_field]
        Y_dict[key] = [float(row[field]) if row[field] else None]


np.random.shuffle(keys)
num_lines = len(keys)
X_train = np.array([X_dict[key] for key in keys[int(num_lines*0.2):]])
Y_train = np.array([Y_dict[key] for key in keys[int(num_lines*0.2):]])
X_valid = np.array([X_dict[key] for key in keys[:int(num_lines*0.2)]])
Y_valid = np.array([Y_dict[key] for key in keys[:int(num_lines*0.2)]])

# means of non-None value
nums = np.sum(np.where(np.not_equal(X_train, None), np.ones_like(X_train), np.zeros_like(X_train)), axis=0)
sums = np.sum(np.where(np.not_equal(X_train, None), X_train, np.zeros_like(X_train)), axis=0)
means = sums/nums

# insert means
X_train = np.where(np.not_equal(X_train, None), X_train, means)
X_valid = np.where(np.not_equal(X_valid, None), X_valid, means)

# pre-process data because values range too much
std_ds_x = np.std(X_train, axis=0, dtype=np.float32)
std_ds_x = np.where(np.greater(std_ds_x, 0.1), std_ds_x, 0.1)
means_x = np.mean(X_train, axis=0, dtype=np.float32)
means_y = np.mean(Y_train, dtype=np.float32)

X_train = (X_train - means_x)/std_ds_x
X_valid = (X_valid - means_x)/std_ds_x

Y_train = Y_train - means_y
Y_valid = Y_valid - means_y

in_dim = X_train.shape[1]
out_dim = 1
batch_size = 32

training_data_generator = BatchGenerator(X_train, Y_train, batch_size)
valid_data_generator = BatchGenerator(X_valid, Y_valid, batch_size)


X = tf.placeholder(tf.float32, [batch_size, in_dim])
Y = tf.placeholder(tf.float32, [batch_size, out_dim])
#outputs = tf.layers.dense(X, 100)
outputs = tf.layers.dense(X, out_dim)

'''
# R2 score
Y_mean = tf.reduce_mean(Y)
#print(Y_mean.shape)
total_error = tf.reduce_sum(tf.square(tf.subtract(Y, Y_mean)))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y, outputs)))
#print(total_error.shape, unexplained_error.shape)
loss = tf.subtract(1.0, tf.div(unexplained_error, total_error))
'''

# mse loss
loss = tf.losses.mean_squared_error(Y, outputs)


optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for _ in range(19):
        xs, ys = next(training_data_generator.generate())
        _, l, outs = sess.run([train_step, loss, outputs], feed_dict={X: xs, Y: ys})
        print(l)
    print(outs.reshape(batch_size, )-ys)
    #print(ys + means_y)

