#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : create_dataset_noise.py
# Creation Date : 05-12-2016
# Last Modified : Fri Dec  9 12:26:58 2016
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""
Load ONE .mseed and Benz et al., 2015 catalog and create tfrecords of noise.
The tfrecords are the noise traces and a label -1.
e.g.,
./bin/preprocess/create_dataset_noise.py \
--stream data/streams/GSOK029_2-2014.mseed \
--catalog data/catalog/Benz_catalog.csv\
--output data/tfrecords/GSOK029_2-2014.tfrecords
"""


import os
import numpy as np
from data_pipeline import DataWriter
import tensorflow as tf
import sys
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
sys.path.append('C:\\Users\\Azhang6\\Downloads\\iFindTimeSeriesData_test')
from loadData import Profile
from scipy.interpolate import interp1d
import pandas as pd




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
flags.DEFINE_float('window_size', 150,
                   'size of the window samples (in seconds)')
flags.DEFINE_float('window_step', 300,
                   'size of the window step(in seconds)')
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
        
    output_name =  "Noise{}.tfrecords".format(0)
    output_path = os.path.join(FLAGS.output_dir, output_name)
    writer = DataWriter(output_path)
    idx = 0
    count = 1
    for stream_file,label_file in zip(stream_files,label_files):
        path_to_stream_file = os.path.join(FLAGS.stream_dir,stream_file)
        path_to_label_file  = os.path.join(FLAGS.stream_dir,label_file)
        print ("+ Loading {} and {}".format(stream_file,label_file))
        data = np.load(path_to_stream_file)
        label = np.load(path_to_label_file)
        ind = np.where(label ==0)[0]
        
        #print("+ Creating tfrecords for {} events".format(len(ind)))
        data = preprocess_data(data)
        data = data[ind,:,:]
        for nevent in range(len(data)):
            writer.write(data[nevent],int(label[nevent]))
            idx +=1
            if(idx==100):
                idx = 0
                writer.close()
                output_name =  "Noise{}.tfrecords".format(count)
                output_path = os.path.join(FLAGS.output_dir, output_name)
                writer = DataWriter(output_path)
                count +=1
                print("+ Created tfrecords for {} batch".format(count-1))
                
            
     
    ##############################################################################
        
    os.chdir(r"C:\Users\Azhang6\Downloads\iFindTimeSeriesData")    
    dirs = os.listdir()
    
    for file in dirs:
    #print(file)
        os.chdir(r"C:\Users\Azhang6\Downloads\iFindTimeSeriesData")
        if (file.endswith(".zip")):
            print(file)
            outputfilename = file[:-4]
            cc = os.getcwd()
            if (outputfilename not in cc ):
                os.chdir(outputfilename)
            csvfiles = [f for f in os.listdir() if f.endswith('.csv') and os.path.getsize(f) > 1000]
            if(len(csvfiles)==0):
                continue
            filee = csvfiles[0]
            
            print(filee)
            
            p1 = Profile(filee)
            
            ####################Quality control to make sure the csv file is valid
            if (p1.notcomplete == True):
                continue
            if (p1.dt == 0 or p1.dt>=10):
                continue
            if (min(p1.time[1:]-p1.time[:-1])<=0):
                continue
            
            p1.lowPassfil()
            p1.get_termi_time2(min_distance=int(150//p1.dt), threshold_rel=0.5, Tmin = 10)
            interpress = interp1d(p1.time, p1.Tpres, kind='linear')  # Linear interpolation
            interslur = interp1d(p1.time, p1.srate, kind='linear')  # Linear interpolation
            Tb_all = p1.time[p1.nInd]-30
            Te_all = p1.time[p1.nInd]+120
            window_start = p1.time[0]
            window_end = window_start + FLAGS.window_size
            while(window_end<p1.time[-1]):
                after_start = Tb_all < window_start
                before_end = Te_all > window_start
                
                after_start1 = Tb_all < window_end
                before_end1 = Te_all > window_end
                
                if(len(np.where(after_start == before_end)[0])!=0 or len(np.where(after_start1 == before_end1)[0])!=0 ):
                    print("skip")
                else:
                    tnew = np.arange(window_start, window_end, 1)
                    snew = interslur(tnew)
                    pnew = interpress(tnew)
                    data = np.zeros((3,int(FLAGS.window_size)))
                    data[0, :] = tnew[:]
                    data[1, :] = snew[:]
                    data[2, :] = pnew[:]
                    
                    writer.write(data,int(0))
                    idx +=1
                    if(idx==100):
                        idx = 0
                        writer.close()
                        output_name =  "Noise{}.tfrecords".format(count)
                        output_path = os.path.join(FLAGS.output_dir, output_name)
                        writer = DataWriter(output_path)
                        count +=1
                        print("+ Created tfrecords for {} batch".format(count-1))

                    
                window_start += FLAGS.window_step
                window_end +=  FLAGS.window_step
            
                
            

if __name__ == "__main__":
    tf.app.run()
        
        
        
#        for filee in os.listdir():
#            print(filee)
#            if ( filee[-1] != "v" or os.path.getsize(filee) < 1000):
#                continue
#            #filee = "C:\\Users\\Azhang6\\Downloads\\iFindTimeSeriesData\\Baily_Unit_13H\\Baily_Unit_13H.Stage16.csv"
#            p1 = Profile(filee)
#            if (p1.notcomplete == True):
#                continue
#            if (p1.dt == 0 or p1.dt>=10):
#                continue
#            if (min(p1.time[1:]-p1.time[:-1])<=0):
#                continue
#            p1.lowPassfil()
#            p1.get_termi_time2(min_distance=int(150//p1.dt), threshold_rel=0.5, Tmin = 10)
#            interpress = interp1d(p1.time, p1.Tpres, kind='linear')  # Linear interpolation
#            interslur = interp1d(p1.time, p1.srate, kind='linear')  # Linear interpolation
#            
#
#            j =1
#            if(len(p1.nInd)>0):
#                for ele in p1.nInd:
#
#                    time = p1.time[ele]
#                    tb = time - 30
#                    te = time + 120  # we cut a 150s profile to represent the shut in window
#                    tnew = np.arange(tb, te, 1)
#                    pnew = interpress(tnew)
#                # pall.append(pnew)
#                    snew = interslur(tnew)
#                # sall.append(snew)
#                    data[i - 1, 0, :] = tnew[:]
#                    data[i - 1, 1, :] = snew[:]
#                    data[i - 1, 2, :] = pnew[:]
#                    jobname.append(filee)
#                    ax1 = f.add_subplot(len(p1.nInd) // 2 + 1, 2,j)
#                    ax1.plot(tnew, snew)
#                # ax1.plot(p1.time[ele-30:ele+90],p1.srate[ele-30:ele+90])
#                # ax1.plot(p1.time[:-1],p1.srate[:-1]-p1.srate[1:])
#                    ax1.set_ylabel('Slurry Rate')
#                    ax2 = ax1.twinx()
#                    ax2.plot(tnew, pnew, 'r')
#                # ax2.plot(p1.time[ele-30:ele+90],p1.Tpres[ele-30:ele+90],'r')
#                # plt.xlim(p1.time[p1.termi_time-10],p1.time[p1.tend+10])
#                    ax2.set_ylabel('Pressure', color='r')
#                    ax2.tick_params('y', colors='r')
#                    ax1.set_xlabel('Elaspe Time (s)')
#
#                    if (i == batch):
#                        i = 0
#                        np.save(r"C:\Users\Azhang6\Downloads\iFindTimeSeriesData\batch" + str(ibatch), data)
#                        data = np.zeros((batch, 3, 150))
#                        ibatch += 1
#                        with open(r"C:\Users\Azhang6\Downloads\iFindTimeSeriesData\batch"+str(ibatch)+"name"+".txt",'w') as ff:
#                            for namee in jobname:
#                                ff.write(namee + '\n')
#                        jobname = []
#                    i += 1
#                    j += 1
#            if(j!=1):
#                plt.savefig(filee[:-4] + "Zoom"+".png")
#            plt.close("all")
#        os.chdir("..")
#
#    
#    
#    
#    
#    
#    
#    
############################################################################

## Create dir to store tfrecords
#    if not os.path.exists(FLAGS.output_dir):
#        os.makedirs(FLAGS.output_dir)
#
#    # Load stream
#    stream_path = FLAGS.stream_path
#    stream_file = os.path.split(stream_path)[-1]
#    print "+ Loading Stream {}".format(stream_file)
#    stream = read(stream_path)
#    print '+ Preprocessing stream'
#    stream = preprocess_stream(stream)
#
#    # Dictionary of nb of events per tfrecords
#    metadata = {}
#    output_metadata = os.path.join(FLAGS.output_dir,"metadata.json")
#
#    # Load Catalog
#    print "+ Loading Catalog"
#    cat = load_catalog(FLAGS.catalog)
#    starttime = stream[0].stats.starttime.timestamp
#    endtime = stream[-1].stats.endtime.timestamp
#    print "startime", UTCDateTime(starttime)
#    print "endtime", UTCDateTime(endtime)
#    cat = filter_catalog(cat, starttime, endtime)
#    print "First event in filtered catalog", cat.Date.values[0], cat.Time.values[0]
#    print "Last event in filtered catalog", cat.Date.values[-1], cat.Time.values[-1]
#    cat_event_times = cat.utc_timestamp.values
#
#    # Write event waveforms and cluster_id=-1 in .tfrecords
#    n_tfrecords = 0
#    output_name = "noise_" + stream_file.split(".mseed")[0] + \
#                  "_" + str(n_tfrecords) + ".tfrecords"
#    output_path = os.path.join(FLAGS.output_dir, output_name)
#    writer = DataWriter(output_path)
#
#    # Create window generator
#    win_gen = stream.slide(window_length=FLAGS.window_size,
#                           step=FLAGS.window_step,
#                           include_partial_windows=False)
#    if FLAGS.max_windows is None:
#        total_time = stream[0].stats.endtime - stream[0].stats.starttime
#        max_windows = (total_time - FLAGS.window_size) / FLAGS.window_step
#    else:
#        max_windows = FLAGS.max_windows
#
#    # Create adjacent windows in the stream. Check there is no event inside
#    # using the catalog and then write in a tfrecords with label=-1
#
#
#    n_tfrecords = 0
#    for idx, win in enumerate(win_gen):
#
#        # If there is not trace skip this waveform
#        n_traces = len(win)
#        if n_traces == 0:
#            continue
#        # Check trace is complete
#        if len(win)==3:
#            n_samples = min(len(win[0].data),len(win[1].data))
#            n_samples = min(n_samples, len(win[2].data))
#        else:
#            n_sample = 10
#        n_pts = win[0].stats.sampling_rate * FLAGS.window_size + 1
#        # Check if there is an event in the window
#        window_start = win[0].stats.starttime.timestamp
#        window_end = win[-1].stats.endtime.timestamp
#        after_start = cat_event_times > window_start
#        before_end = cat_event_times < window_end
#        try:
#            cat_idx = np.where(after_start == before_end)[0][0]
#            event_time = cat_event_times[cat_idx]
#            is_event = True
#            assert window_start < cat.utc_timestamp.values[cat_idx]
#            assert window_end > cat.utc_timestamp.values[cat_idx]
#            print "avoiding event {}, {}".format(cat.Date.values[cat_idx],
#                                                 cat.Time.values[cat_idx])
#        except IndexError:
#            # there is no event
#            is_event = False
#            if (len(win)==3) and (n_pts == n_samples):
#                # Write tfrecords
#                writer.write(win,-1)
#                # Plot events
#                if FLAGS.plot:
#                    trace = win[0]
#                    viz_dir = os.path.join(
#                        FLAGS.output_dir, "viz", stream_file.split(".mseed")[0])
#                    if not os.path.exists(viz_dir):
#                        os.makedirs(viz_dir)
#                    trace.plot(outfile=os.path.join(viz_dir,
#                                                    "noise_{}.png".format(idx)))
#        if idx % 1000  ==0 and idx != 0:
#            print "{} windows created".format(idx)
#            # Save num windows created in metadata
#            metadata[output_name] = writer._written
#            print "creating a new tfrecords"
#            n_tfrecords +=1
#            output_name = "noise_" + stream_file.split(".mseed")[0] + \
#                          "_" + str(n_tfrecords) + ".tfrecords"
#            output_path = os.path.join(FLAGS.output_dir, output_name)
#            writer = DataWriter(output_path)
#
#        if idx == max_windows:
#            break
#
#    # Cleanup writer
#    print("Number of windows  written={}".format(writer._written))
#    writer.close()
#
#    # Write metadata
#    metadata[stream_file.split(".mseed")[0]] = writer._written
#    write_json(metadata, output_metadata)
#

