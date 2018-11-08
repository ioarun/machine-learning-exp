import tensorflow as tf
from PIL import Image
from numpy import array
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import csv
import time
from random import randint

data_path = 'data/'

train_batch_size =  64
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
number_out = 3 # eef pose x, y, z


counter = 0
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


def create_arrays(dictionary, pt):
	# cube_pose_x = dictionary['cube_pose_x'][pt]
	# cube_pose_y = dictionary['cube_pose_y'][pt]
	# list_y = []
	# for i in range(0, 64, 2):
	# 	list_y.append(cube_pose_x)
	# 	list_y.append(cube_pose_y)
	y = []

	r_config = []

	for i in range(0, 7):
		r_config.append(dictionary['right_j'+str(i)][pt])
		# y.append(dictionary['right_j'+str(i)+'_next'][pt])
	y.append(dictionary['eef_pose_x'])
	y.append(dictionary['eef_pose_y'])
	y.append(dictionary['eef_pose_z'])

	return array(y), array(r_config)

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
def sample_data(dict, counter):
	# data_pts = random.sample(range(1, 5863), 64)
	if counter >= 5800:
		counter = 0
	data_pts = []
	for i in range(counter, counter+train_batch_size):
		data_pts.append(i)
	
	counter += 63
	# print (data_pts)
	random.shuffle(data_pts)
	
	img_arr = []
	y_arr = []
	robot_config_arr = []
	count = 0
	for pt in data_pts:

		if count == 0:
			# read an image 
			img = Image.open("data/"+str(pt)+".jpeg")
			img_arr.append(array(img))
			y, r_config = create_arrays(dictionary, pt)
			y_arr.append(y)
			robot_config_arr.append(r_config)
			count = 1
			continue 

		# read an image 
		img = Image.open("data/"+str(pt)+".jpeg")
		img_arr.append(array(img))
		y, r_config = create_arrays(dictionary, pt)
		y_arr.append(y)
		robot_config_arr.append(r_config)

	return img_arr, y_arr, robot_config_arr, counter

def plotter(prediction, input_img):
	list_val =  prediction[0][0]
	x_s = []
	y_s = []
	for i in range(0, len(list_val), 2):
		x_s.append(list_val[i])
		y_s.append(list_val[i+1])

	arr_img = array(input_img)
	# print x_s, y_s
	plt.figure(1)
	plt.subplot(2, 1, 1)
	plt.imshow(arr_img)

	plt.figure(2)
	# plt.subplot(2, 1, 2)
	plt.plot(y_s, x_s, 'ro')
	plt.axis([0, 0.9135, 0, 0.9135])
	plt.show()





x = tf.placeholder(tf.float32, shape=[train_batch_size, img_size, img_size, num_channels], name='x')
robot_config = tf.placeholder(tf.float32, shape=[train_batch_size, number_robot_config], name='robot_config')
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

features_with_robot_config = tf.concat([feature_keypoints, robot_config], -1)


# fully connected layer 1
layer_fc1 = new_fc_layer(input=features_with_robot_config,
						num_inputs=number_features+number_robot_config,
						num_outputs=fc_size,
						use_relu=True)

# fully connected layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
						num_inputs=fc_size,
						num_outputs=fc_size,
						use_relu=True)

# fully connected layer 3
layer_fc3 = new_fc_layer(input=layer_fc2,
						num_inputs=fc_size,
						num_outputs=number_out,
						use_relu=False)


# y_truth
y_true = tf.placeholder(tf.float32, shape=[train_batch_size, number_out], name='y_true')

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=feature_keypoints, labels=y_true))
# cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_true, layer_fc3)))
cost = tf.sqrt(tf.losses.mean_squared_error(layer_fc3, y_true))
'''
cost = tf.losses.mean_squared_error(y_true,
								    layer_fc1,
								    weights=1.0,
								    scope=None,
								    loss_collection=tf.GraphKeys.LOSSES,
								    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
								)
'''
optimizer = tf.train.AdamOptimizer(learning_rate=0.02).minimize(cost)

saver = tf.train.Saver()


sess = tf.Session()

# sess.run(tf.global_variables_initializer())
saver.restore(sess, "models/model.ckpt")


dictionary = csv_file_to_list()

ctr = 0


def optimize(num_iterations):
	global total_iterations
	global ctr
	start_time = time.time()
	for i in range(total_iterations, total_iterations + num_iterations):
		counter = ctr
		x_batch, y_true_batch, robot_config_, ctr = sample_data(dictionary, counter)
		feed_dict_train = {x: x_batch, y_true: y_true_batch, robot_config: robot_config_}
		o, fc = sess.run([optimizer, layer_fc3], feed_dict=feed_dict_train)
		# print (fc, y_true_batch)
		# return 
		# print status after every 50 iterations
		if i % 100 == 0:
			save_path = saver.save(sess, "models/model.ckpt")
			cos = sess.run(cost, feed_dict=feed_dict_train)
			print ("cost :", cos)

	total_iterations += num_iterations

	# ending time
	end_time = time.time()
	time_diff = end_time - start_time


optimize(500000)

