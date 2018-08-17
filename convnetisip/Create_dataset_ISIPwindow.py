# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:00:07 2018

@author: AZhang6
"""

import os
import numpy as np
from data_pipeline import DataWriter
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)


flags = tf.flags
flags.DEFINE_string('stream_dir','C:\\Users\\Azhang6\\Downloads\\iFindTimeSeriesData',
                    'path to the directory of ISIP time series.')
flags.DEFINE_string('catalog', None, 'path to the labels of detected windows.')
flags.DEFINE_string('output_dir','C:\\Users\\Azhang6\\Downloads\\TFrec',
                    'path to the directory in which the tfrecords are saved')
flags.DEFINE_bool("plot", False,
                  "If we want the event traces to be plotted")

FLAGS = flags.FLAGS

def preprocess_data(data):
#For pressure data, we do maxmin normalization
#For Slurry rate, we do maxabs normalization
    for i in range(len(data)):
        data[i,1,:] = MaxAbsScaler().fit_transform(X=data[i,1,:].reshape(-1, 1)).flatten()
        data[i,2,:] = MinMaxScaler().fit_transform(X=data[i,2,:].reshape(-1, 1)).flatten() 
    return(data)
    
    
def main(_):
    stream_files = ["batch"+str(j)+".npy" for j in range(1273)]  #We have 1273 small batches
    label_files = ["batch"+str(j)+"_ISIPlabel.npy" for j in range(1273)]
    print (str(len(stream_files))+" batches to import")
    
    # Create dir to store tfrecords
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    for stream_file,label_file in zip(stream_files,label_files):
        path_to_stream_file = os.path.join(FLAGS.stream_dir,stream_file)
        path_to_label_file  = os.path.join(FLAGS.stream_dir,label_file)
        print ("+ Loading {} and {}".format(stream_file,label_file))
        data = np.load(path_to_stream_file)
        label = np.load(path_to_label_file)
        ind = np.where(label ==1)[0]
        
        output_name = stream_file.split(".npy")[0] + ".tfrecords"
        output_path = os.path.join(FLAGS.output_dir, output_name)
        writer = DataWriter(output_path)

        

        print("+ Creating tfrecords for {} events".format(len(ind)))
        data = preprocess_data(data)
        data = data[ind,:,:]
        label = label[ind]
        for nevent in range(len(data)):
            writer.write(data[nevent],int(label[nevent])) 
        writer.close()
    
    
    

if __name__ == "__main__":
    tf.app.run()
