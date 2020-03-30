# Filename: Hect3dcnn.py
# Brief: 4 convolutional layers followed by 2 fully-connect layers
# By: Yijie Xu
# Environments:
# Python 3
# Tensorflow 1.12
# 方案1，方案2都可以通过设置函数hectConv3d的输入参数kernel_type来调整
# 详情请看hectConv3d处的注释

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

# DATASET_ROOT = None
# MODEL_DIR = None
device = 'GPU'

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NUM = 10
SHUFFLE_SIZE = 400
rate = 0

def cnn_model_fn(features, labels, mode):

    LEARNING_RATE = 0.005 / (10**rate)
    HET_KERNEL_TYPE = 1

    """
    kernel_size的值对应的卷积核大小
    kernel_type | kernel shape
     0          | (H, W, D) (1, 1, 1)
     1          | (H, W, 1) (1, 1, D)
     2          | (H, W, D) (1, 1, D)

    具体的网络层结构
    layer name    | kernel shape
    conv1(3d)     | 3*5*5
    pool1         | 2*2
    conv2(het3d)  | -> kernel_type
    pool2         | 3*3
    conv3(het3d)  | -> kernel_type
    pool3         | 3*3
    conv4(het3d)  | -> kernel_type
    pool4         | 3*3
    flatten
    """

    """Model function for CNN."""
    # Input Layer
    input_layer = tf.cast(tf.reshape(features['snippet'], [-1, 5, 10, 240, 320]), dtype=tf.float32)

    # Convolutional Layer #1
    conv1 =  tf.layers.conv3d(
           inputs=input_layer,
           filters=6,
           kernel_size=[3, 5, 5],
           padding="valid",
           strides=1,
           activation=tf.nn.relu,
           data_format='channels_first',
           name='3dconv1')


    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(inputs=conv1,
                                    pool_size=[1, 2, 2],
                                    strides=(1, 2, 2),
                                    data_format='channels_first')

    # Convolutional Layer #2 and Polling Layer #2
    hectconv2 = hectConv3d(inputs=pool1,
              input_channel=6,
              output_channel=8,
              kernel_size=[3, 5, 5],
              kernel_type=HET_KERNEL_TYPE)

    pool2 = tf.layers.max_pooling3d(inputs=hectconv2,
                                    pool_size=[1, 3, 3],
                                    strides=(1, 3, 3),
                                    data_format='channels_first')

    # Convolutional Layer #3 and Pooling Layer #3
    hectconv3 = hectConv3d(inputs=pool2,
              input_channel=8,
              output_channel=16,
              kernel_size=[3, 5, 5],
              kernel_type=HET_KERNEL_TYPE)

    pool3 = tf.layers.max_pooling3d(inputs=hectconv3,
                                    pool_size=[1, 3, 3],
                                    strides=(1, 3, 3),
                                    data_format='channels_first')

    # Convolutional Layer #3 and Pooling Layer #3
    hectconv4 = hectConv3d(inputs=pool3,
              input_channel=16,
              output_channel=6,
              kernel_size=[3, 5, 5],
              kernel_type=HET_KERNEL_TYPE)

    pool4 = tf.layers.max_pooling3d(inputs=hectconv4,
                                    pool_size=[1, 3, 3],
                                    strides=(1, 3, 3),
                                    data_format='channels_first')

    # Dense Layer
    pool4_flat = tf.layers.flatten(pool4)


    dense1 = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
            inputs=dense1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=CLASS_NUM)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the
        # 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #print("---logits", logits.shape.dims)
    #print("---labels", labels.shape.dims)
    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('loss', loss)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
                 loss=loss,
                 global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                                 mode=mode,
                                 loss=loss,
                                 train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(
                   labels=labels,
                   predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
                                 mode=mode,
                                 loss=loss,
                                 eval_metric_ops=eval_metric_ops)


def _parse_function(example_proto):
    """Convert str-encoding data to 4-dims tensor.
       (channel, depth, height, width)"""
    image_feature_description = {
        'snippet' : tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, image_feature_description)

    snippet = parsed_features['snippet']
    snippet_arr = tf.io.decode_raw(snippet, np.uint8)
    snippet_arr = tf.reshape(snippet_arr, (5, 10, 240, 320))
    label = parsed_features['label']

    return {'snippet': snippet_arr}, label


def train_input_fn(filenames, batch_size=16):
    """Input function for training."""
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(SHUFFLE_SIZE, reshuffle_each_iteration=False)
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def eval_input_fn(filenames):
    """Input function for evaluation"""
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(SHUFFLE_SIZE, reshuffle_each_iteration=False)
    dataset = dataset.repeat().batch(16)
    return dataset

# 方案1，方案2都可以通过设置函数hectConv3d的输入参数kernel_type来调整
def hectConv3d(inputs, input_channel, output_channel, kernel_size, kernel_type=2):
    """
    Hect convolution layer.
    可以通过设置参数kernel_type来选择方案，具体的对应关系如下所示：
    (H 卷积核的高，W 卷积核的宽，D 卷积核的深度)
    kernel_type | kernel shape
     0          | (H, W, D) (1, 1, 1) 时间异构卷积核方案1
     1          | (H, W, 1) (1, 1, D) 时间异构卷积核方案2
     2          | (H, W, D) (1, 1, D) 时间同构的卷积核
    """
    inputs = tf.cast(inputs, tf.float32)
    split_kernel = [1] * input_channel
    channels = tf.split(inputs, split_kernel, 1)

    ori_kernel_size = list(kernel_size)
    point_kernel_size = list(kernel_size)
    point_kernel_size[1] = 1
    point_kernel_size[2] = 1

    # 根据不同的kernel_type配置
    # 不同的point_kernel_size和ori_kernel_size
    if kernel_type == 0:
        point_kernel_size[0] = 1
    elif kernel_type == 1:
        ori_kernel_size[0] = 1
    elif kernel_type == 2:
        pass
    else:
        print("invalid kernel type")

    layers = []

    for flag in range(output_channel):

        slices = []

        for i in range(input_channel):

            c = channels[i]

            kernel_size = None
            use_bias = False

            if i == 0:
                  use_bias = True

            kernel_size = point_kernel_size if i != flag%input_channel else ori_kernel_size

            conv1 = tf.layers.conv3d(
                  inputs=c,
                  filters=1,
                  kernel_size=kernel_size,
                  padding="same",
                  strides=1,
                  use_bias = use_bias,
                  data_format='channels_first')

            slices.append(conv1)

        stacked = tf.stack(slices)
        stacked = tf.reduce_mean(stacked, 0)
        stacked = tf.nn.relu(stacked)
        stacked = tf.squeeze(stacked, 1)
        layers.append(stacked)

    hectcnn_layer = tf.stack(layers, 1)
    return hectcnn_layer

def shuffle_tfrecord_list(file_list):
    n = len(file_list)
    cp_file_list = list(file_list)
    for i in range(500):
        src = random.randint(0, n-1)
        dst = random.randint(0, n-1)
        tmp = cp_file_list[dst]
        cp_file_list[dst] = cp_file_list[src]
        cp_file_list[src] = tmp
    return cp_file_list


def main():
    # 1000 steps will roughly cover 1 epoch of data.
    save_step_num = 10
    training_steps = 500
    evaluating_steps = 200

    eval_record_path = '/content/drive/My Drive/UCF10_test'
    train_record_path = '/content/drive/My Drive/UCF10_train'

    train_batch_size = 16

    class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking',
                   'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving',
                   'Drumming', 'Fencing']

    assert len(class_names) == CLASS_NUM

    # Get Training tfrecords
    train_filenames = []
    for cn in class_names:
        re_str = str(os.path.join(train_record_path, cn)) + '/*.tfrecord'
        train_filenames += glob.glob(re_str)

    # Get Evaluating tfrecords
    eval_filenames = []
    for cn in class_names:
        re_str = str(os.path.join(eval_record_path, cn)) + '/*.tfrecord'
        eval_filenames += glob.glob(re_str)

    train_filenames = shuffle_tfrecord_list(train_filenames)

    for f in train_filenames:
        print(f)

    config = tf.estimator.RunConfig(
                       model_dir="/content/drive/My Drive/hect_3dcnn_1_4conv_0402",
                       save_summary_steps=save_step_num,
                       log_step_count_steps=20)

    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
                       model_fn=cnn_model_fn,
                       config=config)

    for i in range(20):
        
        # Train classifier
        cnn_classifier.train(
                   input_fn=lambda:train_input_fn(train_filenames, train_batch_size),
                   steps=training_steps)
        # Evaluation
        eval_results = cnn_classifier.evaluate(
                   input_fn=lambda:eval_input_fn(eval_filenames),
                   steps=evaluating_steps)
        print(eval_results)

        train_filenames = shuffle_tfrecord_list(train_filenames)

        for f in train_filenames:
            print(f)


if __name__ == '__main__':
    main()
