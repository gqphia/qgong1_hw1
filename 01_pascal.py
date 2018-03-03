from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial
import cv2
import matplotlib.pyplot as plt

import pdb
from eval import compute_map
#import models

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
] # 20 classes


def cnn_model_fn(features, labels, mode, num_classes=20):
    # the same model from MNIST
    # Input Layer, the shape is:[batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # Convolutional Layer #1, the output shape is [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1, the output shape is [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5], # or kernel_size=5
        padding="same",     # the value of padding: "valid"(default), "same"
        activation=tf.nn.relu)

    # Pooling Layer #2, the output shape is [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # flatten the feature map pool2 to shape: [batch_size, features] 
    pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 64])

    # Dense Layer (fc layer?) and apply dropout regularization
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)  #
    dropout = tf.layers.dropout(    #rate: dropout rate, dropout only in TRAIN
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer. The raw values for our predictions.
    # create a dense layer with 20 nrutons(20 classes) with linear activation
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        # instead of uisng softmax, here use sigmoid
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')
    
    
    #accuracy= tf.metrics.accuracy(
    #        labels=tf.argmax(labels,axis=1), predictions=predictions["classes"])
    #tf.summary.scalar("accuracy",accuracy[1])


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # for Tensorboard
        tf.summary.scalar("training_loss", loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
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
    print("##### Start to Load %s Data ... ####" % split)
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
        #print("labels %s:" % classi)
        #print(labels)
        #print("weights %s:" % classi)
        #print(weights)

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
        model_dir="./tmp/pascal_model_scratch")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # save training loss


    # Train for 1000 iterations, and save once for every NUM_ITERS iters
    train_iters = 1000 #1000
    log_iters = 25 #10
    NUM_ITERS = train_iters/log_iters
    BATCH_SIZE = 100

    #session = tf.InteractiveSession()
    #session.run(tf.global_variables_initializer())
    #write_op = tf.summary.merge_all()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    #session = tf.Session()

    # tf.train.SummaryWriter -> tf.summary.FileWriter
    train_writer = tf.summary.FileWriter("./tmp/pascal_model_scratch")#, session.graph)
    #x = np.multiply(range(log_iters+1), NUM_ITERS)
    #train_loss = np.multiply(range(log_iters+1), 0.0)
    #acc = np.multiply(range(log_iters+1), 0.0)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data, "w": train_weights},
            y=train_labels,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True)

# Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

    # Train the model
    for i in range(log_iters):
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS,
            hooks=[logging_hook])
        #save training loss
        #train_loss[i+1] = logging_hook.tensors.loss
        #print('Training loss at iter %s: %s' % (i*NUM_ITERS, logging_hook.tensors.loss))

        
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

        # store mAP for each iter
        #acc[i+1] = np.mean(AP)
        summary = tf.Summary(value=[tf.Summary.Value(tag='mean_AP',
                                                     simple_value=np.mean(AP))])
        train_writer.add_summary(summary, i*NUM_ITERS)
        print('Accuracy at iter %s: %s' % (i*NUM_ITERS, np.mean(AP)))
        
        #summary1 = session.run(write_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        #train_writer.add_summary(summary1, i*NUM_ITERS)
        #train_writer.flush()


        # plot figure
        #plt.clf()
        #fig = plt.figure(1)
        #plt.plot(x, acc)
        #fig.savefig("Test_mAP_for_Task_1.png")

        #plt.figure(2)
        #plt.plot(x, train_loss)
        #fig.savefig("training loss for Task 1.png")


    train_writer.close()
 

if __name__ == "__main__":
    main()
