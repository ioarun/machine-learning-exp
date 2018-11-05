'''
[[-9.9498749e-01  9.9500370e-01  2.9512200e-01  9.5897859e-01
   1.6487867e-01  9.8282254e-01  7.7839756e-01 -8.4554803e-01
  -9.2326492e-01  9.2326117e-01  5.6504276e-02  3.6453849e-01
   9.9226958e-01  7.7679586e-01  9.8332721e-01  8.3029008e-01
   6.7962760e-01 -5.0879091e-01  8.0734082e-02  3.1335104e-01
  -8.5496765e-01 -8.6031860e-01  8.8034433e-01 -2.0021659e-01
   9.8930991e-01  9.4566905e-01 -9.8997486e-01  9.7342521e-01
  -8.5938203e-01  9.1553599e-01  9.7813070e-02 -9.9418956e-01
   2.3370929e-01  5.3301430e-01 -9.2173524e-02 -9.6556145e-01
   6.1789501e-02  3.9005482e-01  2.3536563e-01  7.9031640e-01
   1.4077574e-01  2.5366151e-01 -9.9999952e-01  1.8213619e-04
  -9.5057197e-02 -1.0000010e+00  1.1603745e-01  9.9683195e-01
  -9.9851120e-01 -7.9045457e-01  9.7150081e-01  3.6361593e-01
   9.7797704e-01 -5.6274092e-01 -9.7238725e-01  9.8190796e-01
   4.4537785e-05 -7.5376593e-06 -9.6490723e-01  8.6958498e-01
   5.3527409e-01 -3.6000326e-02  9.7591239e-01  9.9857712e-01]]

robot_config = 

[-0.7163434229	
-1.048085536	
0.7940572682	
2.9145453115	
1.1799879494	
-0.3952533917	
1.9949735534]	

'''

import tensorflow as tf
from tensorflow.losses import Reduction
from PIL import Image
from numpy import array
from numpy import ndarray
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import time
from random import randint

data_path = 'data/'

train_batch_size = 50
total_iterations = 0

# actual image dimension is 800x800
img_size = 100
# convert it into an input vector of 800*800*800 = 1920000 dimension
img_size_flat = img_size * img_size * img_size

# number of channels
num_channels = 3

# conv 1
filter_size1 = 7      # 7x7 filter dimension
num_filters1 = 32     # 64 filters in layer 1
stride1 = 2

# conv 2
filter_size2 = 5
num_filters2 = 32
stride2 = 1

# conv 3
filter_size3 = 5
num_filters3 = 32
stride3 = 1

number_features = 64
number_robot_config = 7
fc_size = 40
number_out = 7

# tensorflow computation graph begins

# helper functions for creating new weights and biases
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, stride, use_pooling=True):

	# shape of the each filter-weights
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# create new weights
	weights = new_weights(shape=shape)

	# create new biases
	biases = new_biases(length=num_filters)

	# convolution operation
	# strides = 1, padding = 1 (to maintain spatial size same as previous layer)
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,stride,stride,1], padding='SAME')

	# add biases to the results of convolution to each filter
	layer += biases

	if use_pooling:
		# 2x2 max-pooling
		layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# ReLU operation, max(x, 0)
	layer = tf.nn.relu(layer)

	return layer, weights

# flatten layer for fully connected neural net
def flatten_layer(layer):
	# get shape of the input layer
	layer_shape = layer.get_shape()

	# layer shape is of the form [num_images, img_height, img_width, num_channels]
	# num_features = img_height*img_width*num_channels
	num_features = layer_shape[1:4].num_elements()

	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

# create fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# weights and biases for fc layer
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# linear operation
	layer = tf.matmul(input, weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)
	
	return layer


def create_true_y_array(dictionary, pt):
	cube_pose_x = dictionary['cube_pose_x'][pt]
	cube_pose_y = dictionary['cube_pose_y'][pt]
	list_y = []
	for i in range(0, 64, 2):
		list_y.append(cube_pose_x)
		list_y.append(cube_pose_y)


	return array(list_y)

# returns a dictionary from a csv file
def csv_file_to_list():
	# open the file in universal line ending mode 
	with open('data/0.csv', 'rU') as infile:
	 # read the file as a dictionary for each row ({header : value})
	  reader = csv.DictReader(infile)
	  data = {}
	  for row in reader:
	    for header, value in row.items():
	      try:
	        data[header].append(value)
	      except KeyError:
	        data[header] = [value]
	return data

# returns x_batch, y_truth
def sample_data(dict):
	data_pts = random.sample(range(1, 2020), 50)
	img_arr = []
	y_arr = []
	count = 0
	for pt in data_pts:

		if count == 0:
			# read an image 
			img = Image.open("data/"+str(pt)+".jpeg")
			img_arr.append(array(img))
			y_arr.append(array(create_true_y_array(dictionary, pt)))
			count = 1
			continue 

		# read an image 
		img = Image.open("data/"+str(pt)+".jpeg")
		img_arr.append(array(img))
		y_arr.append(create_true_y_array(dictionary, pt))

	return img_arr, y_arr

x = tf.placeholder(tf.float32, shape=[train_batch_size, img_size, img_size, num_channels], name='x')
robot_config = tf.placeholder(tf.float32, shape=[None, number_robot_config], name='robot_config')
# conv layers require image to be in shape [num_images, img_height, img_weight, num_channels]
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# conv layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
											num_input_channels=num_channels,
											filter_size=filter_size1,
											num_filters=num_filters1,
											stride=stride1,
											use_pooling=False)
# conv layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
											num_input_channels=num_filters2,
											filter_size=filter_size2,
											num_filters=num_filters2,
											stride=stride2,
											use_pooling=False)

# conv layer 3
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
											num_input_channels=num_filters3,
											filter_size=filter_size3,
											num_filters=num_filters3,
											stride=stride3,
											use_pooling=False)

# spatial softmax layer
feature_keypoints = tf.contrib.layers.spatial_softmax(layer_conv3,
											temperature=None,
    										name=None,
    										variables_collections=None,
    										trainable=True,
    										data_format='NHWC')

# features_with_robot_config = tf.concat([feature_keypoints, robot_config], -1)


# fully connected layer 1
# layer_fc1 = new_fc_layer(input=features_with_robot_config,
# 						num_inputs=number_features+number_robot_config,
# 						num_outputs=fc_size,
# 						use_relu=True)

# # fully connected layer 2
# layer_fc2 = new_fc_layer(input=layer_fc1,
# 						num_inputs=fc_size,
# 						num_outputs=number_out,
# 						use_relu=True)

layer_fc1 = new_fc_layer(input=feature_keypoints,
						num_inputs=number_features,
						num_outputs=64,
						use_relu=False)


# y_truth
y_true = tf.placeholder(tf.float32, shape=[train_batch_size, 64], name='y_true')

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=feature_keypoints, labels=y_true))
# cost = tf.reduce_mean(tf.squared_difference(y_true, feature_keypoints))
cost = tf.losses.mean_squared_error(y_true,
								    layer_fc1,
								    weights=1.0,
								    scope=None,
								    loss_collection=tf.GraphKeys.LOSSES,
								    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
								)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)

correct_predictions = tf.equal(layer_fc1, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

saver = tf.train.Saver()


sess = tf.Session()

sess.run(tf.global_variables_initializer())
# saver.restore(sess, "models/model.ckpt")


dictionary = csv_file_to_list()


def optimize(num_iterations):
	global total_iterations
	start_time = time.time()
	for i in range(total_iterations, total_iterations + num_iterations):
		x_batch, y_true_batch = sample_data(dictionary)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}
		o, fc = sess.run([optimizer,layer_fc1], feed_dict=feed_dict_train)
		# print fc, y_true_batch 
		# return
		# print status after every 100 iterations
		if i % 50 == 0:
			save_path = saver.save(sess, "models/model.ckpt")
			acc = sess.run(accuracy, feed_dict=feed_dict_train)
			cos = sess.run(cost, feed_dict=feed_dict_train)
			print "cost :", cos

	total_iterations += num_iterations

	# ending time
	end_time = time.time()
	time_diff = end_time - start_time


optimize(10000)
# read an image 
# img = Image.open("image1.jpeg")
# arr1 = array(img)
# # read an image 
# img = Image.open("image2.jpeg")
# arr2 = array(img)

# list_img = []
# list_img.append(arr1)
# list_img.append(arr2)

# # arr = np.concatenate((arr1, arr2))
# # arr = np.concatenate((arr, arr1))

# print list_img

# print arr1 , '\n'
# print arr2 , '\n'
# print sess.run(feature_keypoints, feed_dict={x: arr.reshape(2, -1)})

# feature1 = list(features[0])
# _robot_config = [[-0.7163434229, -1.048085536, 0.7940572682, 2.9145453115, 1.1799879494, -0.3952533917, 1.9949735534]]


# print (sess.run(layer_fc2, feed_dict={x: arr.reshape(1, -1), robot_config: _robot_config}))

