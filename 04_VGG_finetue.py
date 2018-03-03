from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import scipy
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import cv2
import matplotlib.pyplot as plt
import os

from eval import compute_map

# import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
	'aeroplane',
	'bicycle',
	'bird',
	'boat',
	'bottle',
	'bus',
	'car',
	'cat',
	'chair',
	'cow',
	'diningtable',
	'dog',
	'horse',
	'motorbike',
	'person',
	'pottedplant',
	'sheep',
	'sofa',
	'train',
	'tvmonitor',
]


def cnn_model_fn(features, labels, mode, num_classes=20):
	# Write this function
	"""Model function for CNN."""
	# Input Layer
	# N = features["x"].shape[0]
	input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

	# Data Augmentation
	# Train: Random crops and left-right flips
	if mode == tf.estimator.ModeKeys.TRAIN:
		tmp = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer)
		tmp = tf.map_fn(lambda img: tf.random_crop(img, size=[224,224,3]), tmp)
		augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[224,224]), tmp)
		# add to tensorboard
		tf.summary.image('training_images', augment_input)
	# Test: Center crop
	elif mode == tf.estimator.ModeKeys.PREDICT:
		tmp = tf.map_fn(lambda img: tf.image.central_crop(img, central_fraction=0.8), input_layer)
		augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[224,224]), tmp)

	# consistent with the pre-trained model
	with tf.variable_scope('vgg_16') as scope:
		with tf.variable_scope("conv1"):
			conv1 = tf.layers.conv2d(
				inputs=augment_input,
				kernel_size=[3, 3],
				strides=1,
				filters=64,
				padding="same",
				activation=tf.nn.relu,
				name = "conv1_1")

			# convolution layer #2: conv3-64
			conv2 = tf.layers.conv2d(
				inputs=conv1,
				kernel_size=[3, 3],
				strides=1,
				filters=64,
				padding="same",
				activation=tf.nn.relu,
				name = "conv1_2")

			# max 1
			pool1 = tf.layers.max_pooling2d(
				inputs=conv2,
				pool_size=[2, 2],
				strides=2,
				name = "pool1")

		##########################
		# 3
		with tf.variable_scope("conv2"):
			conv3 = tf.layers.conv2d(
				inputs=pool1,
				kernel_size=[3, 3],
				strides=1,
				filters=128,
				padding="same",
				activation=tf.nn.relu,
				name = "conv2_1")
			# convolution layer #4: conv3-128
			conv4 = tf.layers.conv2d(
				inputs=conv3,
				kernel_size=[3, 3],
				strides=1,
				filters=128,
				padding="same",
				activation=tf.nn.relu,
				name = "conv2_2")
			# max 2
			pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

		##########################
		# 5
		with tf.variable_scope("conv3"):
			conv5 = tf.layers.conv2d(
				inputs=pool2,
				kernel_size=[3, 3],
				strides=1,
				filters=256,
				padding="same",
				activation=tf.nn.relu,
				name = "conv3_1")
			# convolution layer #6: conv3-256
			conv6 = tf.layers.conv2d(
				inputs=conv5,
				kernel_size=[3, 3],
				strides=1,
				filters=256,
				padding="same",
				activation=tf.nn.relu,
				name = "conv3_2")
			# convolution layer #7: conv3-256
			conv7 = tf.layers.conv2d(
				inputs=conv6,
				kernel_size=[3, 3],
				strides=1,
				filters=256,
				padding="same",
				activation=tf.nn.relu,
				name = "conv3_3")
			# max 2
			pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

		# convolution layer #8: conv3-512
		with tf.variable_scope("conv4"):
			conv8 = tf.layers.conv2d(
				inputs=pool3,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv4_1")
			# convolution layer #9: conv3-512
			conv9 = tf.layers.conv2d(
				inputs=conv8,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv4_2")
			# convolution layer #10: conv3-512
			conv10 = tf.layers.conv2d(
				inputs=conv9,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv4_3")
			# max 2
			pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

		# convolution layer #11: conv3-512 
		with tf.variable_scope("conv5"):
			conv11 = tf.layers.conv2d(
				inputs=pool4,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv5_1")
			# convolution layer #12: conv3-512
			conv12 = tf.layers.conv2d(
				inputs=conv11,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv5_2")
			# convolution layer #13: conv3-512
			conv13 = tf.layers.conv2d(
				inputs=conv12,
				kernel_size=[3, 3],
				strides=1,
				filters=512,
				padding="same",
				activation=tf.nn.relu,
				name = "conv5_3")
			# max 2
			pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)

		#with tf.variable_scope("flatten"):
		pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])

		# fully_connected(4096)
		# relu()
		#with tf.variable_scope("fc6"):
		dense1 = tf.layers.dense(inputs=pool5_flat, units=4096,
									 activation=tf.nn.relu, name="fc6")

		#with tf.variable_scope("dropout1"):
		dropout1 = tf.layers.dropout(
			inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

		# fully_connected(4096)
		#with tf.variable_scope("fc7"):
		dense2 = tf.layers.dense(inputs=dropout1, units=4096,
									 activation=tf.nn.relu, name="fc7")

		# dropout(0.5)
		# with tf.variable_scope("dropout2"):
		dropout2 = tf.layers.dropout(
			inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

		# fully_connected(20)
		# Logits Layer
		#with tf.variable_scope("fc8"):
		logits = tf.layers.dense(inputs=dropout2, units=20, 
									kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
									bias_initializer=tf.zeros_initializer(),
									name="fc8")

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		#"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.identity(tf.losses.sigmoid_cross_entropy(
		multi_class_labels=labels, logits=logits), name='loss')

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		tf.summary.scalar("training_loss", loss)

		#exclude = ['vgg_16/fc8/kernel', 'vgg_16/fc8/bias']
		variables_to_restore = tf.contrib.framework.get_trainable_variables()
		variables_to_restore = variables_to_restore[:-2]
		#variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude = exclude)
		#variables_to_restore = variables_to_restore[:-2]
		scopes = {os.path.dirname(v.name) for v in variables_to_restore}
		tf.train.init_from_checkpoint('new_vgg_16.ckpt', 
									  {s + '/': s + '/' for s in scopes})
		#################################

		decayed_learning_rate = tf.train.exponential_decay(
			0.0001,  # Base learing rate
			global_step=tf.train.get_global_step(),
			decay_steps=1000,  # Decay step DONT MODIFY!!
			decay_rate=0.5,    # Decay rate
			staircase=True)
		# add lr to tensorboard
		tf.summary.scalar('learning_rate', decayed_learning_rate)

		# SGD + Momentum optimizer
		optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate,
											   momentum=0.9)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())

		# plot histogram of gradients
		train_summary = []
		grads_and_vars = optimizer.compute_gradients(loss)
		# tf.summary.histogram("grad_histogram",grads_and_vars)
		for g, v in grads_and_vars:
			if g is not None:
				# print(format(v.name))
				grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name[:-2]), g)
				train_summary.append(grad_hist_summary)
		tf.summary.merge(train_summary)

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=train_op)

	# EVAL mode
	tf.summary.scalar('test_loss', loss)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["probabilities"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
	# return tf.estimator.EstimatorSpec(
	#     mode=mode, loss=loss)

def load_pascal(data_dir, split='train'):
	"""
	Function to read images from PASCAL data folder.
	Args:
		data_dir (str): Path to the VOC2007 directory.
		split (str): train/val/trainval split to use.
	Returns:
		images (np.ndarray): Return a np.float32 array of
			shape (N, H, W, 3), where H, W are 224px each,
			and each image is in RGB format.
		labels (np.ndarray): An array of shape (N, 20) of
			type np.int32, with 0s and 1s; 1s for classes that
			are active in that image.
		weights: (np.ndarray): An array of shape (N, 20) of
			type np.int32, with 0s and 1s; 1s for classes that
			are confidently labeled and 0s for classes that
			are ambiguous.
	"""
	# Write this function
	print("##### TASK4: Start to Load %s Data ... ####" % split)
	H = 256
	W = 256

	filename = data_dir + "/ImageSets/Main/" + split +".txt"
	img_dir = data_dir + "/JPEGImages/";
	with open(filename) as f:
		filelist = f.read().splitlines()
	num_imgs = len(filelist)
	print("num of %s imgs: %s" % (split, num_imgs))
	# read images
	imgs = np.zeros([num_imgs, H, W, 3], np.float32)
	count = 0
	for i in filelist: #range(num_imgs):
		img = Image.open(img_dir + i + '.jpg')
		img = img.resize((W, H), Image.ANTIALIAS)
		imgs[count, :, :, :] = img
		count = count + 1

	# read labels and weights
	labels = np.zeros([num_imgs, 20]).astype(int)
	weights = np.ones([num_imgs, 20]).astype(int)
	for classi in range(20):
		class_file = data_dir + "/ImageSets/Main/" \
						 + CLASS_NAMES[classi] + "_" + split + ".txt"
		with open(class_file) as f:
			#clslist = f.read().splitlines()
			clslist = f.readlines()
		clslist = [x.split() for x in clslist]
		for i in range(num_imgs):
			labels[i, classi] = int(int(clslist[i][1])==1)
			weights[i, classi] = int(int(clslist[i][1])!=0)

	print("##### Data Loaded ####")
	return imgs, labels, weights


def parse_args():
	parser = argparse.ArgumentParser(
		description='Train a classifier in tensorflow!')
	parser.add_argument(
		'data_dir', type=str, default='data/VOC2007',
		help='Path to PASCAL data storage')
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args


def _get_el(arr, i):
	try:
		return arr[i]
	except IndexError:
		return arr

def load_vgg_pretrained_model(ckpt_path):
	restore_model = dict() #{}
	# load model
	reader = tf.train.NewCheckpointReader(ckpt_path)
	#mean_rgb = reader.get_tensor("vgg_16/mean_rgb")
	for tensor_name in reader.get_variable_to_shape_map():
		#print("tensor name: ", old_name)
		tensor = reader.get_tensor(tensor_name)
		#print("tensor value: ", tensor)
		tensor_var = tf.Variable(tensor)

		# convert fc6,fc7(4d in the pretrained model) to 2d
		if (tensor_name == "vgg_16/fc6/weights"):
			tensor_var = tf.Variable(tf.reshape(tensor_var, [7 * 7 * 512, 4096]))
		elif (tensor_name == "vgg_16/fc7/weights"):
			tensor_var = tf.Variable(tf.reshape(tensor_var, [4096, 4096]))

		# transform "weights" and "biases" in .ckpt to "kernel" and "bias"(tf.layers.conv2d)
		new_tensor_name = tensor_name.replace("weights", "kernel").replace("biases", "bias")
		restore_model[new_tensor_name] = tensor_var

	saver = tf.train.Saver(restore_model)

	# save to the updated vgg model
	with tf.Session() as sess:
		print("###### Start to load vgg16 pretrained model ... #####")
		sess.run(tf.global_variables_initializer()) #tf.global_variables_initializer())
		saver.save(sess, './new_vgg_16.ckpt')
		print("###### vgg16 pretrained model loaded ########")



def main():
	args = parse_args()
	# Load training and eval data
	train_data, train_labels, train_weights = load_pascal(
		args.data_dir, split='trainval')
	eval_data, eval_labels, eval_weights = load_pascal(
		args.data_dir, split='test')

	pascal_classifier = tf.estimator.Estimator(
		model_fn=partial(cnn_model_fn,
						 num_classes=train_labels.shape[1]),
		model_dir="./tmp/vgg_model_scratch_finetune_1")
	tensors_to_log = {"loss": "loss"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=100)
	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data, "w": train_weights},
		y=train_labels,
		batch_size=10,
		num_epochs=None,
		shuffle=True)

	# draw
	total_iters = 4000 
	iter = 10 # PLOT 10 points is ok
	NUM_ITERS = int(total_iters / iter)
	mAP_writer = tf.summary.FileWriter("./tmp/vgg_model_scratch_finetune_1")

	# load pretrained vgg model
	load_vgg_pretrained_model("vgg_16.ckpt")

	for i in range(iter):
		pascal_classifier.train(
			steps=NUM_ITERS,
			hooks=[logging_hook],
			input_fn=train_input_fn)
		# Evaluate the model and print results
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": eval_data, "w": eval_weights},
			y=eval_labels,
			num_epochs=1,
			shuffle=False)
		pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
		pred = np.stack([p['probabilities'] for p in pred])
		rand_AP = compute_map(
			eval_labels, np.random.random(eval_labels.shape),
			eval_weights, average=None)
		print('Random AP: {} mAP'.format(np.mean(rand_AP)))
		gt_AP = compute_map(
			eval_labels, eval_labels, eval_weights, average=None)
		print('GT AP: {} mAP'.format(np.mean(gt_AP)))
		AP = compute_map(eval_labels, pred, eval_weights, average=None)
		print('Obtained {} mAP'.format(np.mean(AP)))
		print('per class:')
		for cid, cname in enumerate(CLASS_NAMES):
			print('{}: {}'.format(cname, _get_el(AP, cid)))

		# add test loss to tensorboard
		#ev = pascal_classifier.evaluate(input_fn=eval_input_fn)
		#summary0 = tf.Summary(value=[tf.Summary.Value(tag='test_loss',
		#											 simple_value=ev["loss"])])
		#mAP_writer.add_summary(summary0, i*NUM_ITERS)

		# draw graph
		print('accuracy: %d' % np.mean(AP))
		summary = tf.Summary(value=[tf.Summary.Value(tag='test_mAP',
													 simple_value=np.mean(AP))])
		mAP_writer.add_summary(summary, i*NUM_ITERS)
		print('Accuracy at iter %s: %s' % (i*NUM_ITERS, np.mean(AP)))

	mAP_writer.close()

if __name__ == "__main__":
	main()
