# save_to_TFRecords.py
# 
# Save pictures as TFRecord file
# 
# By: Yijie Xu
# Last change: March 17 2019

from __future__  import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
import os
import cv2
import glob

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double"""
    return tf.train.Feature(float_liat=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns a int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    
def serialize_example(snippet, lab):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    feature = {
      'snippet': _bytes_feature(snippet),
      'label': _int64_feature(lab) 
    }
  
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0, f1):
    """
    Transfrom to a scalar tensor.
    """
    tf_string = tf.py_func(
              serialize_example, 
              (f0, f1),  # pass these args to the above function.
              tf.string) # the return type is <a href="../../api_docs/python/
    return tf.reshape(tf_string, ()) # The result is a scalar


def extract_snippet(filelist_dict, snippet_length):
    """
    Extract snippet from 10 frames (5 xframe, 5 yframe)
    # gray     (snippet_length) 
    # grad_x   (snippet_length)
    # grad_y   (snippet_length)
    # mbh_x    (snippet_length)
    # mbh_y    (snippet_length)
    """
    assert len(filelist_dict['gray']) == snippet_length
    assert len(filelist_dict['grad_x']) == snippet_length
    assert len(filelist_dict['grad_y']) == snippet_length
    assert len(filelist_dict['mbh_x']) == snippet_length
    assert len(filelist_dict['mbh_y']) == snippet_length
    
    snippet = bytearray()
    
    # gray 
    for f in filelist_dict['gray']:
        image = cv2.imread(f, 0)
        image_string = image.tostring()             
        snippet.extend(image_string)

    # gradient x
    for f in filelist_dict['grad_x']:  
        image = cv2.imread(f, 0)
        image_string = image.tostring()             
        snippet.extend(image_string)

    # gradient y
    for f in filelist_dict['grad_y']:
        image = cv2.imread(f, 0)
        image_string = image.tostring()             
        snippet.extend(image_string)

    # optical flow x
    for f in filelist_dict['mbh_x']:  
        image = cv2.imread(f, 0)
        image_string = image.tostring()             
        snippet.extend(image_string)

    # optical flow y
    for f in filelist_dict['mbh_y']:  
        image = cv2.imread(f, 0)
        image_string = image.tostring()             
        snippet.extend(image_string)
        
    return bytes(snippet)


def parse_class_folder_to_tfrecord(gray_path,
                                   grad_class_path, 
                                   mbh_class_path, 
                                   lab):
    """
    Parse a class folder and to transfrom all images in every
    video folder into a TFRecord.
    """
    _snippet_length = 10
    _gap = 2

    gray_folders = os.listdir(gray_path)
    grad_folders = os.listdir(grad_class_path)
    mbh_folders = os.listdir(mbh_class_path)

    assert len(grad_folders) == len(mbh_folders)   
    assert len(grad_folders) == len(gray_folders) 
    video_num = len(grad_folders)

    gray_folder_paths = [os.path.join(gray_path, x) for x in gray_folders] 
    grad_folder_paths = [os.path.join(grad_class_path, x) for x in grad_folders]
    mbh_folder_paths = [os.path.join(mbh_class_path, x) for x in grad_folders]
    
    snippet_num = 0    

    for i in range(video_num):  
        videoname = os.path.splitext(grad_folders[i])[0]
        filename = '{0!s}.tfrecord'.format(videoname)
        writer = tf.data.experimental.TFRecordWriter(filename)
        print("Finding pictures in directory:")
        print(gray_folder_paths[i])
        print(grad_folder_paths[i])
        print(mbh_folder_paths[i])

        snippets_list = []

        # Factories of snippets
        # when there are 10 frames, save to a snippet record.
        gray_frames = []
        x_grad_frames = []
        y_grad_frames = []
        x_mbh_frames = []
        y_mbh_frames = []

        count = 1
        while True:
            count += 1
            
            gray_re_str = gray_folder_paths[i] + '/{}_gray_{:d}.jpg'.format(videoname, count)
            x_grad_re_str = grad_folder_paths[i] + '/{}_diff_{:d}_01.jpg'.format(videoname, count)
            y_grad_re_str = grad_folder_paths[i] + '/{}_diff_{:d}_02.jpg'.format(videoname, count)
            x_mbh_re_str = mbh_folder_paths[i] + '/{}_mbh_{:d}_01.jpg'.format(videoname, count)
            #print(glob.glob(x_mbh_re_str))
            y_mbh_re_str = mbh_folder_paths[i] + '/{}_mbh_{:d}_02.jpg'.format(videoname, count)
            
            # Save a pair frames every _gap pairs of frames 
            if count % _gap == 0:
                gray_frames += glob.glob(gray_re_str)
                x_grad_frames += glob.glob(x_grad_re_str)
                y_grad_frames += glob.glob(y_grad_re_str)
                x_mbh_frames += glob.glob(x_mbh_re_str)
                y_mbh_frames += glob.glob(y_mbh_re_str)
            else:
                continue

            if len(glob.glob(gray_re_str)) < 1 or\
               len(glob.glob(x_grad_re_str)) < 1 or\
               len(glob.glob(y_grad_re_str)) < 1 or\
               len(glob.glob(x_mbh_re_str)) < 1 or\
               len(glob.glob(y_mbh_re_str)) < 1:
                break
            elif len(gray_frames) == _snippet_length:
                frames_dict = {'gray': gray_frames, 
                               'grad_x': x_grad_frames,
                               'grad_y': y_grad_frames,
                               'mbh_x': x_mbh_frames,
                               'mbh_y': y_mbh_frames}
                print(count)
                snippet = extract_snippet(frames_dict, _snippet_length) 
                snippets_list.append(snippet)  
                # Clear factories  
                gray_frames = []
                x_grad_frames = []
                y_grad_frames = []
                x_mbh_frames = []
                y_mbh_frames = []
               
        
        labs = [lab] * len(snippets_list)
        snippet_num += len(snippets_list)

        dataset = tf.data.Dataset.from_tensor_slices((snippets_list, labs))

        serialized_dataset = dataset.map(tf_serialize_example)
        writer.write(serialized_dataset)
        print(filename, "saved.")

    return snippet_num


class cd:
    """Change current directory in a safe way."""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        
def main():


    gray_path = "/mnt/ehpc-hz-JdjdHW8IZe/UCF50_gray/"
    grad_path = "/mnt/ehpc-hz-JdjdHW8IZe/UCF50_diff/"
    mbh_path = "/mnt/ehpc-hz-JdjdHW8IZe/UCF50_mbh/"
    output_path = "/mnt/ehpc-hz-JdjdHW8IZe/UCF_Dataset_tfrecord/"

    #class_names = os.listdir('/mnt/UCF50_videos/UCF50/')
    class_names = os.listdir(gray_path)
    class_names.sort()
    # Print class informations.
    for cn in class_names:
        print(cn)
    n = len(class_names)
    print(n, " classes")

    # Dictionary to fetch the numeric label of a class.
    class2label = dict(zip(class_names, range(n)))
   
    for i in range(10):
    #for cn in class_names:
        if i < 2:
            continue
        cn = class_names[i]
        # Label
        label = class2label[cn]
        # Paths
        gray_class_path = os.path.join(gray_path, cn)
        grad_class_path = os.path.join(grad_path, cn)
        mbh_class_path = os.path.join(mbh_path, cn)
        output_class_path = os.path.join(output_path, cn)

        if not os.path.isdir(output_class_path):
            os.mkdir(output_class_path)

        with cd(output_class_path):
            parse_class_folder_to_tfrecord(gray_class_path, 
                                           grad_class_path,
                                           mbh_class_path,
                                           label)
    
if __name__ == '__main__':
    main()
