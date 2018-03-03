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
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # Data Augmentation
    # Train: Random crops and left-right flips
    if mode == tf.estimator.ModeKeys.TRAIN:
        tmp = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_layer)
        tmp = tf.map_fn(lambda img: tf.random_crop(img, size=[224,224,3]), tmp)
        augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[256,256]), tmp)
        # add to tensorboard
        tf.summary.image('training_images', augment_input)
    # Test: Center crop
    elif mode == tf.estimator.ModeKeys.PREDICT:
        tmp = tf.map_fn(lambda img: tf.image.central_crop(img, central_fraction=0.8), input_layer)
        augment_input = tf.map_fn(lambda img: tf.image.resize_images(img, size=[256,256]), tmp)


    # add Network Graph to tensorboard
    # convolution layer #1: conv3-64
    with tf.variable_scope('conv1'):
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
    with tf.variable_scope('pool1'):
        # pooling layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name = "pool1")

    # convolution layer #3: conv3-128
    with tf.variable_scope('conv2'):
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

    with tf.variable_scope('pool2'):
        # pooling layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name = "pool2")

    # convolution layer #5: conv3-256
    with tf.variable_scope('conv3'):
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

    with tf.variable_scope('pool3'):
        # pooling layer #3
        pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2, name = "pool3")

    # convolution layer #8: conv3-512
    with tf.variable_scope('conv4'):
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
    with tf.variable_scope('pool4'):
        # pooling layer #4
        pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2, name = "pool4")

    # convolution layer #11: conv3-512
    with tf.variable_scope('conv5'):
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
    with tf.variable_scope('pool5'):
        # pooling layer #5
        pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2, name = "pool5")

    # flatten
    pool5_flat = tf.reshape(pool5, [-1, 8 * 8 * 512])
    # fc(4096)
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096,
                            activation=tf.nn.relu, name = "fc6")
    # dropout
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # fc(4096)
    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu, name = "fc7")
    # dropout
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20, name = "fc8")

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

        decayed_learning_rate = tf.train.exponential_decay(
            0.001,  # Base learing rate
            global_step=tf.train.get_global_step(),
            decay_steps=100,  # Decay step
            decay_rate=0.5,    # Decay rate
            staircase=True)
        # add lr to tensorboard
        tf.summary.scalar('learning_rate', decayed_learning_rate)

        # SGD + Momentum optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate,
                                               momentum = 0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        # add histogram of gradients to tensorboard
        train_summary =[]
        grads_and_vars = optimizer.compute_gradients(loss)
        # tf.summary.histogram("grad_histogram",grads_and_vars)
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name[:-2]), g)
                #sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                train_summary.append(grad_hist_summary)
                #train_summary.append(sparsity_summary)
        tf.summary.merge(train_summary)

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    tf.summary.scalar('test_loss', loss)
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
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    print("##### TASK3: Start to Load %s Data ... ####" % split)
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
        model_dir="./tmp/vgg_model_scratch")
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

    total_iters = 40000 # report first 2 hours(would take 5~6 hours to train)
    iter = 100
    NUM_ITERS = int(total_iters/iter)
    mAP_writer = tf.summary.FileWriter("./tmp/vgg_model_scratch")#,sess.graph)

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