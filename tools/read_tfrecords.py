# read_TFRecords.py
#
# Read data from TFRecord files
#
# By: Yijie Xu
# Last change: April 3 2019
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 320, 240, 10])
    plt.imshow(input_layer[:,:,:,0])
    plt.xlabel(labels[0])
    plt.show()

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

    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the
        # 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                 loss=loss,
                 global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(
                   labels=labels, predictions=prdictions["classes"])}
    return tf.estimator.EstimatorSpec(
               mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _parse_function(example_proto):
    image_feature_description = {
        'snippet' : tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, image_feature_description)

    snippet = parsed_features['snippet']
    snippet_arr = tf.io.decode_raw(snippet, np.uint8)
    snippet_arr = tf.reshape(snippet_arr, (10, 240, 320))
    label = parsed_features['label']
    #plt.plot(snippet_arr[0,:,:])
    return snippet_arr, label

#def bytes_to_numpy()

def main():
    record_root_path = '/mnt/ehpc-hz-JdjdHW8IZe/UCF3_op_tfrecord/'
    class_names = ['BaseballPitch']

    for cn in class_names:
        re_str = str(os.path.join(record_root_path, cn)) + '/*.tfrecord'
        filenames = glob.glob(re_str)
    print(filenames)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset.batch(1).repeat(1)
    dataset.shuffle(1000, reshuffle_each_iteration=False)

    DATASET_SIZE = 1439
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    def train_input_fn():
        """An input function for training"""
        return train_dataset.make_one_shot_iterator().get_next()

    def test_input_fn():
        """An input function for training"""
        return test_dataset.make_one_shot_iterator().get_next()

    def val_input_fn():
        """An input function for training"""
        return train_dataset.make_one_shot_iterator().get_next()

    my_feature_columns = [tf.feature_column.numeric_column(key='snippet')]

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
                     model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
                 tensors=tensors_to_log, every_n_iter=50)

    mnist_classifier.train(
                   input_fn=train_input_fn,
                   hooks=[logging_hook])

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    main()
