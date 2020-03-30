import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

device = 'GPU'
#tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

    LEARNING_RATE = 0.08
    CLASS_NUM = 10

    """Model function for CNN."""
    # Input Layer
    input_layer = tf.cast(tf.reshape(features['snippet'], [-1, 5, 10, 240, 320]), dtype=tf.float32)

    # Convolutional Layer #1
    conv1 =  tf.layers.conv3d(
           inputs=input_layer,
           filters=6,
           kernel_size=[3, 7, 7],
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
    conv2 = tf.layers.conv3d(
          inputs=pool1,
          filters=16,
          kernel_size=[3, 6, 8],
          padding="valid",
          strides=1,
          activation=tf.nn.relu,
          data_format='channels_first')  

    pool2 = tf.layers.max_pooling3d(inputs=conv2, 
                                    pool_size=[1, 3, 3], 
                                    strides=(1, 3, 3),
                                    data_format='channels_first')
    
    # Convolutional Layer #2 and Polling Layer #2
    conv3 = tf.layers.conv3d(
          inputs=pool2,
          filters=16,
          kernel_size=[3, 6, 8],
          padding="valid",
          strides=1,
          activation=tf.nn.relu,
          data_format='channels_first')  

    pool3 = tf.layers.max_pooling3d(inputs=conv3, 
                                    pool_size=[1, 3, 3], 
                                    strides=(1, 3, 3),
                                    data_format='channels_first')
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv4 = tf.layers.conv3d(
          inputs=pool3,
          filters=6,
          kernel_size=[3, 5, 5],
          padding="valid",
          strides=1,
          activation=tf.nn.relu,
          data_format='channels_first')
    
    pool4 = tf.layers.max_pooling3d(inputs=conv4, 
                                    pool_size=[1, 3, 3], 
                                    strides=(1, 3, 3),
                                    data_format='channels_first')

    # Dense Layer
    pool4_flat = tf.layers.flatten(pool4)

    dense1 = tf.layers.dense(inputs=pool4_flat, units=512, activation=tf.nn.relu)
    
    dropout1 = tf.layers.dropout(
            inputs=dense1, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=128, activation=tf.nn.relu)
   
    dropout2 = tf.layers.dropout(
            inputs=dense2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    
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

    # Calculate Loss (for both TRAIN and EVAL modes)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    acc, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])

    tf.summary.scalar('accuracy', acc)
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
    dataset = dataset.shuffle(200, reshuffle_each_iteration=False)
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def eval_input_fn(filenames):
    """Input function for evaluation"""
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat().batch(32) 
    return dataset

  
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
    save_step_num = 10
    train_record_path = '/content/drive/My Drive/UCF10_train'
    eval_record_path = '/content/drive/My Drive/UCF10_test'

    train_batch_size = 16
    class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking',
                   'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving',
                   'Drumming', 'Fencing']

    train_filenames = []
    
    for cn in class_names:
        re_str = str(os.path.join(train_record_path, cn)) + '/*.tfrecord'
        train_filenames += glob.glob(re_str)
     
    train_filenames = shuffle_tfrecord_list(train_filenames)
    
    # Get Eval tfrecords
    eval_filenames = []

    for cn in class_names:
        re_str = str(os.path.join(eval_record_path, cn)) + '/*.tfrecord'
        eval_filenames += glob.glob(re_str)
        
    config = tf.estimator.RunConfig(
                       model_dir="/content/drive/My Drive/3dcnn_04048",
                       save_summary_steps=save_step_num,
                       log_step_count_steps=20)
    
    # Create the Estimator
    cnn_classifier = tf.estimator.Estimator(
                       model_fn=cnn_model_fn, 
                       config=config)
    
    for i in range(10):
        # Train classifier
        cnn_classifier.train(
                   input_fn=lambda:train_input_fn(train_filenames, train_batch_size),
                   steps=1000)

        eval_results = cnn_classifier.evaluate(
                   input_fn=lambda:eval_input_fn(eval_filenames),
                   steps=200)
        print(eval_results)

if __name__ == '__main__':
    main()  

