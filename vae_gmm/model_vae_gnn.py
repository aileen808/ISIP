from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
from skimage import io
import data_pipeline as dp
from ops import *
from utils import *
from sklearn import mixture
from sklearn.cluster import KMeans
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
class VAE(object):
    def __init__(self, sess, image_size=150,
                 batch_size=100, sample_size=100, output_size=150,
                 z_dim=2, c_dim=2, class_size = 5, dataset='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            image_size: The size of input image.
            batch_size: The size of batch. Should be specified before training.
            sample_size: (optional) The size of sampling. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [28]
            z_dim: (optional) Dimension of latent vectors. [5]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.class_size = class_size
        #self.n_epochs = 
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.win_size = image_size
        self.build_model()

    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # TODO: Define encoder network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            # The output of encoder network should have two parts:
            # A mean vector and a log(std) vector. Both of them have
            # the same dimension with latent vector z.
            #######################################################
            net = lrelu(conv1d(image,32,4,2,name='conv1'));
            net = lrelu(batch_norm(conv1d(net, 32, 4, 2,name = 'conv2'),train=train,name='bc1'))
            net = lrelu(batch_norm(conv1d(net, 32, 4, 2,name = 'conv3'),train=train,name='bc2'))
            net = lrelu(batch_norm(conv1d(net, 32, 4, 2,name = 'conv4'),train=train,name='bc3'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(batch_norm(linear(net, 256,scope='l1'),train=train,name='bc4'))
            net = lrelu(batch_norm(linear(net, 128,scope='l2'),train=train,name='bc5'))
            net = lrelu(batch_norm(linear(net, 64,scope='l3'),train=train,name='bc6'))
            gaussian_params = linear(net, 2 * self.z_dim,scope='l4')
            #with tf.name_scope()
            self.gaussian_params = gaussian_params
            z_mean = gaussian_params[:, :self.z_dim]
            z_log_std = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])
            with tf.name_scope("latent"):
                tf.summary.histogram("mu",z_mean)
                tf.summary.histogram("sigma",z_log_std)
        
            return(z_mean,z_log_std)
            #######################################################
            #                   end of your code
            #######################################################
    def classifier(self,z,reuse= False,train = True):
        with tf.variable_scope("classifier",reuse = reuse):
            k = self.class_size
            net = tf.nn.relu(batch_norm(linear(z,32,scope='cla1'),train = train,name = 'clas1'))
            net = tf.nn.relu(batch_norm(linear(net,64,scope='cla2'),train=train,name = 'clas2'))
            net = linear(net,k,scope ='cla3')
            self.cat = tf.nn.softmax(net,name = 'predicted_class')
            return net,self.cat
            
            
            
    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. To make the
            # output pixel values in [0,1], add a sigmoid layer before
            # the output. Also use batch normalization layer after
            # deconv layer, and use 'train' argument to indicate the
            # mode of bn layer. Note that when sampling images using
            # trained model, you need to set train='False'.
            #######################################################
            net = tf.nn.relu(batch_norm(linear(z, 64,scope='del1'),train=train,name='deb1'))
            net = tf.nn.relu(batch_norm(linear(z, 128,scope='del3'),train=train,name='deb3'))
            net = tf.nn.relu(batch_norm(linear(z, 256,scope='del4'),train=train,name='deb4'))
            net = tf.nn.relu(batch_norm(linear(net, 10 * 32,scope='del2'),train=train,name='deb2'))
            net = tf.reshape(net, [self.batch_size, 10, 32])
            net = tf.nn.relu(batch_norm(deconv1d(net, [self.batch_size, 19, 32], 4, 2,name='de5'), name='debn5',train =train))
            net = tf.nn.relu(batch_norm(deconv1d(net, [self.batch_size, 38, 32], 4, 2,name='de4'), name='debn4',train =train))
            net = tf.nn.relu(batch_norm(deconv1d(net, [self.batch_size, 75, 32], 4, 2,name='de2'), name='debn3',train =train))
            out = deconv1d(net, [self.batch_size, 150, 2], 4, 2,name='de3')
            return out
            #######################################################
            #                   end of your code
            #######################################################

    def kl_cat(self):
        #prior = self.theta_p.reshape((1,-1))
        prior = tf.ones([self.batch_size,self.class_size],dtype = tf.float32)
        #prior = tf.reshape(tf.tile(prior, self.batch_size), [self.batch_size, tf.shape(prior[0]])
        posterior = self.cat
        klcat = tf.reduce_sum((posterior * tf.log(posterior / (prior + 1e-12)) + 1e-12),[1])
        return(tf.reduce_mean(klcat))
        
    def kl_gauss(self,mu,logvar,mu_c,logvar_c):
        klg = tf.reduce_sum(0.5 * (logvar_c - logvar + tf.exp(logvar) / (tf.exp(logvar_c) + 1e-8) +
                      (mu - mu_c)**2 / (tf.exp(logvar_c) + 1e-8) - 1),[1])
        return (tf.reduce_mean(klg))
    def predict(self,config,train = False):
        

        print(config.win_size)
        data_path = os.path.join(config.dataset,"signal")
        print(data_path)
        train_pipeline = dp.DataPipeline(data_path, config, True)
        print(train_pipeline)
        train_batch = train_pipeline.samples
        val_path = os.path.join(config.dataset,"signal_test")
        val_pipeline = dp.DataPipeline(val_path, config, True)
        val_batch = val_pipeline.samples
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        for epoch in xrange(config.epoch):
            print(epoch)
            train_data = self.sess.run(train_pipeline.samples)
           # print([epoch,1])
            batch_z =np.random.normal(0, 1, (config.batch_size, self.z_dim)).astype(np.float32) 
            batch_z = np.hstack([batch_z,np.ones((config.batch_size, self.class_size))/self.class_size])

            #fig = plt.figure()
            test_data = self.sess.run(val_pipeline.samples)
            yobs = test_data.astype(np.float32)
            #ypred = self.sess.run(self.out,feed_dict={self.inputs:train_data})
            hidden,cat = self.sess.run([self.z,self.cat],feed_dict={self.inputs: train_data})
            col = ['red','green','blue','yellow','black']
            for i in range(self.batch_size):
                idx = np.argmax(cat[i])
                plt.figure(1)
                plt.plot(hidden[i,0],hidden[i,1], color=col[idx],marker = '.')
                plt.figure(idx+2)
                plt.plot(train_data[i,:,0],color='b')
                plt.figure(idx+2)
                plt.plot(train_data[i,:,1],color='r')
                
                        
    
    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################
        bs = self.batch_size
        n_centroid = self.class_size
        #image_dms = [self.input_height, self.input_width, self.c_dim]
        self.inputs =  tf.placeholder(tf.float32,[bs]+[self.image_size,self.c_dim],name = 'real_images')
        #print(tf.shape(self.inputs))
        self.z = tf.placeholder(tf.float32,[bs,self.z_dim+self.class_size],name='z')
        self.cat = tf.placeholder(tf.float32,[bs,self.class_size],name='cat')
        
        #qy_logit, qy = qy_graph(xb, k)
        
        self.mu,self.z_log_var = self.encoder(self.inputs, reuse=False)
        z_log_var = self.z_log_var
        #encoding
        latent = tf.concat([self.mu,self.z_log_var],1) 
        eps = tf.random_normal(tf.shape(self.mu),0,1,dtype=tf.float32)
        
        
        #um = self.unsupervised_model.partial_fit(latent,steps = 1)
        
        cat_logit, cat = self.classifier(latent, reuse=False)
        z =  tf.concat([self.mu+tf.exp(z_log_var/2)*eps,cat],1)
        self.z = tf.concat([self.mu+tf.exp(z_log_var/2)*eps,cat],1)
        
        self.cat = cat
        
        cat_oh = gumbel_softmax(cat, 0.3, hard=False) #onehot catagory
        sums = tf.reduce_sum(cat_oh, axis=0)
        sums_exp_dims = tf.expand_dims(sums, axis=-1)
        p_mu_logvar_c = tf.matmul(tf.transpose(cat_oh,[1,0]), latent)/sums_exp_dims
        p_mu_logvar =  tf.matmul(cat_oh,p_mu_logvar_c)
        kl_cat = self.kl_cat()
        kl_z = self.kl_gauss(self.mu,self.z_log_var,p_mu_logvar[:, :self.z_dim], p_mu_logvar[:, self.z_dim:])
        

        #decoding
        out=self.decoder(z,reuse=False)
        self.out = tf.clip_by_value(out,1e-8, 1 - 1e-8)
        with tf.name_scope('loss'):
        #loss
            mse = tf.reduce_mean(tf.reduce_sum((self.inputs-self.out)**2,[1,2]))
            self.loss = mse + kl_z + kl_cat
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('MSE',mse)
        self.merged = tf.summary.merge_all()
        #######################################################
        #                   end of your code
        #######################################################
        self.saver = tf.train.Saver()

    def train(self, config):
        """Train VAE"""
        # load MNIST dataset
        
        #config.win_size = 150
        print(config.win_size)
        data_path = os.path.join(config.dataset,"signal")
        print(data_path)
        train_pipeline = dp.DataPipeline(data_path, config, True)
        print(train_pipeline)
        train_batch = train_pipeline.samples
        val_path = os.path.join(config.dataset,"signal_test")
        val_pipeline = dp.DataPipeline(val_path, config, True)
        val_batch = val_pipeline.samples
#        #self.unsupervised_model = tf.contrib.learn.KMeansClustering(self.class_szie,
#                                                                    distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE, 
#                                                                    initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT)
                                                                        
        
        
#        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#        data = mnist.train.images
#        data = data.astype(np.float32)
#        data_len = data.shape[0]
#        data = np.reshape(data, [-1, 28, 28, 1])

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_dir = os.path.join(config.sample_dir, config.dataset)
        if not os.path.exists(config.sample_dir):
            os.mkdir(config.sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        print("before_loop")
        self.sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord) 
        writer = tf.summary.FileWriter('logs',self.sess.graph)
        for epoch in xrange(config.epoch):
            #print(epoch)
            train_data = self.sess.run(train_pipeline.samples)
           # print([epoch,1])
            batch_z =np.random.normal(0, 1, (config.batch_size, self.z_dim)).astype(np.float32) 
            #print([epoch,2])
            batch_z_val =np.random.normal(0, 1, (config.batch_size, self.z_dim)).astype(np.float32) 
            #print([epoch,3])
            _,loss,summary = self.sess.run([optim,self.loss,self.merged], feed_dict={self.inputs: train_data})
            writer.add_summary(summary,epoch)
            print("Epoch: [%2d]  time: %4.4f, loss: %.8f" \
                      % (epoch, time.time() - start_time, loss))
            if np.mod(epoch, 10) == 0:
                    self.save(self.checkpoint_dir, epoch)
                    fig = plt.figure()
                    test_data = self.sess.run(val_pipeline.samples)
                    yobs = test_data.astype(np.float32)
                    ypred = self.sess.run(self.out,feed_dict={self.inputs: test_data})
                    nll_lossp,summaryp = self.sess.run([self.loss,self.merged],feed_dict={self.inputs: test_data})
                    writer.add_summary(summaryp,epoch)
                    #summaryp = 
                    ypred1 = ypred.astype(np.float32)
                    for i in range(9):
                        plt.subplot(3,3,i+1)
                        plt.plot(yobs[i,:,0],'b')
                        plt.plot(yobs[i,:,1],'r')
                        plt.plot(ypred1[i,:,0],'b--')
                        plt.plot(ypred1[i,:,1],'r--')
                    plt.savefig('Epoch'+str(epoch)+'.png')
                    print(nll_lossp)
                    plt.close()
                        
#            batch_idxs = min(data_len, config.train_size) // config.batch_size
#            for idx in xrange(0, batch_idxs):
#                counter += 1
#                batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size, :]
#                batch_z =np.random.normal(0, 1, (config.batch_size, self.z_dim)).astype(np.float32)       #prior.gaussian(self.batch_size, self.z_dim)
#                #######################################################
#                # TODO: Train your model here, print the loss term at
#                # each training step to monitor the training process.
#                # Print reconstructed images and sample images every
#                # config.print_step steps. Sample z from standard normal
#                # distribution for sampling images. You may use function
#                # save_images in utils.py to save images.
#                #######################################################
#                #train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
#                #self.sess.run()
#                _,loss, nll_loss, kl_loss = self.sess.run([optim,self.loss,self.neg_loglikelihood, self.KL_div], feed_dict={self.inputs: batch_images, self.z: batch_z})
#                #self.writer.add_summary(summary_str, counter)
#                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
#                      % (epoch, idx,batch_idxs, time.time() - start_time, loss, nll_loss, kl_loss))
#                
#                #######################################################
#                #                   end of your code
#                #######################################################
#                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
#                    self.save(config.checkpoint_dir, counter)

            
    def save(self, checkpoint_dir, step):
        model_name = "isip.model"
        model_dir = "%s_%s_%s" % (self.dataset, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
