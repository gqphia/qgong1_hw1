from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
from os import listdir
import os.path as osp
from PIL import Image
from functools import partial
from math import sqrt
import matplotlib.pyplot as plt
# from skimage import io
import pdb
from eval import compute_map
# import models
import pickle

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

## source from https://gist.github.com/kukuruza/03731dc494603ceab0c5 ##
def visualize_kernel (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


def cnn_model_fn(features, labels, mode, num_classes=20):
   
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # Data Augmentation
    # Train: Random crops and left-right flips
    if mode == tf.estimator.ModeKeys.TRAIN:
        tmp = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer)
        tmp = tf.map_fn(lambda img: tf.random_crop(img, size=[224,224,3]), tmp)
        augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[256,256]), tmp)
    # Test: Center crop
    elif mode == tf.estimator.ModeKeys.PREDICT:
        tmp = tf.map_fn(lambda img: tf.image.central_crop(img, central_fraction=0.8), input_layer)
        augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[256,256]), tmp)

    # AlexNet Implementation
    # Convolutional Layer #1
    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(
            inputs=augment_input,
            filters=96,
            kernel_size=[11, 11],
            strides=(4, 4),
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

        # TASK 5: Visualize conv1 kernels
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('conv2d/kernel')
        grid = visualize_kernel(weights)
        tf.summary.image('conv1/weights', grid, max_outputs=1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        #activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # flatten the feature map pool3 to shape: [batch_size, features] 
    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 256])
    # Fully connected layer #1
    dense1 = tf.layers.dense(
        inputs=pool3_flat, 
        units=4096,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
    # dropout #1
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Fully connected layer #2
    dense2 = tf.layers.dense(
        inputs=dropout1, 
        units=4096,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

    #dropout #2
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer, final fc layer
    logits = tf.layers.dense(inputs=dropout2, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')
    

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar("training_loss", loss)

        decayed_learning_rate = tf.train.exponential_decay(
            0.001,  # Base learing rate
            global_step=tf.train.get_global_step(),
            decay_steps=10000,  # Decay step
            decay_rate=0.5,    # Decay rate
            staircase=True)
        # SGD + Momentum optimizer
        train_op = tf.train.MomentumOptimizer(decayed_learning_rate, 
            momentum=0.9).minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["probabilities"])}
    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
    """
    print("##### TASK2: Start to Load %s Data ... ####" % split)
    H = 256  # height
    W = 256  # weight

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
        model_dir="./tmp/alexnet_model_scratch_visualize")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000) #100
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)

    total_iters = 40000 #40000 
    iter = 20 #50
    NUM_ITERS = int(total_iters/iter)
    # mAP = tf.Variable(0.0, dtype=tf.float32)

    # merged = tf.summary.merge_all()
    mAP_writer = tf.summary.FileWriter("./tmp/alexnet_model_scratch_visualize")

    for i in range(iter):
        
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS,
            hooks=[logging_hook])
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        # pdb.set_trace()
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

        # plot graph
        print('accuracy: %d' % np.mean(AP))
        summary = tf.Summary(value=[tf.Summary.Value(tag='test_mAP',
                                                     simple_value=np.mean(AP))])
        mAP_writer.add_summary(summary, i*NUM_ITERS)
        print('Accuracy at iter %s: %s' % (i*NUM_ITERS, np.mean(AP)))

    mAP_writer.close()


if __name__ == "__main__":
    main()
