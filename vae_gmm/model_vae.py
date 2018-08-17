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
import random
import numpy as np
from matplotlib import pyplot as plt

class VAE(object):
    def __init__(self, sess, image_size=150,
                 batch_size=100, sample_size=100, output_size=150,
                 z_dim=4, c_dim=2, dataset='default',
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
        #image_dms = [self.input_height, self.input_width, self.c_dim]
        self.inputs =  tf.placeholder(tf.float32,[bs]+[self.image_size,self.c_dim],name = 'real_images')
        #print(tf.shape(self.inputs))
        self.z = tf.placeholder(tf.float32,[bs,self.z_dim],name='z')
        self.mu,z_log_var = self.encoder(self.inputs, reuse=False)
        #encoding
        eps = tf.random_normal(tf.shape(self.mu),0,1,dtype=tf.float32)
        z = self.mu+tf.exp(z_log_var/2)*eps
        #decoding
        out=self.decoder(z,reuse=False)
        self.out = tf.clip_by_value(out,1e-8, 1 - 1e-8)
        with tf.name_scope('loss'):
        #loss
            mse = -tf.reduce_sum((self.inputs-self.out)**2,[1,2])
            self.mse = -tf.reduce_mean(mse)
            likelihood = mse
            #likelihood = tf.reduce_sum(self.inputs*tf.log(self.out)+(1-self.inputs)*tf.log(1-self.out),[1,2])  ###OLD 
            self.ce = tf.reduce_sum(self.inputs*tf.log(self.out)+(1-self.inputs)*tf.log(1-self.out),[1,2])  ###OLD 
            #0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
            KL_div = 0.5* tf.reduce_sum(tf.square(self.mu)+tf.exp(z_log_var)-z_log_var-1.,[1])
            self.neg_loglikelihood = -tf.reduce_mean(likelihood)
            self.KL_div = tf.reduce_mean(KL_div)
            ELBO = -self.neg_loglikelihood-self.KL_div
            self.loss = -ELBO
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('MSE',self.mse)
            tf.summary.scalar('nll',self.loss-self.KL_div)
        
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
            _,loss, nll_loss, kl_loss,summary = self.sess.run([optim,self.loss,self.neg_loglikelihood, self.KL_div,self.merged], feed_dict={self.inputs: train_data, self.z: batch_z})
            writer.add_summary(summary,epoch)
            print("Epoch: [%2d]  time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, time.time() - start_time, loss, nll_loss, kl_loss))
            if np.mod(epoch, 10) == 0:
                    self.save(config.checkpoint_dir, epoch)
                    fig = plt.figure()
                    test_data = self.sess.run(val_pipeline.samples)
                    yobs = test_data.astype(np.float32)
                    ypred = self.sess.run(self.out,feed_dict={self.inputs: test_data, self.z: batch_z_val})
                    nll_lossp,summaryp,gg = self.sess.run([self.mse,self.merged,self.gaussian_params],feed_dict={self.inputs: test_data, self.z: batch_z_val})
                    writer.add_summary(summaryp,epoch)
                    #summaryp = 
                    ypred1 = ypred.astype(np.float32)
                    for i in range(9):
                        plt.subplot(3,3,i+1)
                        plt.plot(yobs[i,:,0],'b')
                        plt.plot(yobs[i,:,1],'r')
                        plt.plot(ypred1[i,:,0],'b--')
                        plt.plot(ypred1[i,:,1],'r--')
                        #plt.title(["{0:0.2f}".format(k) for k in gg[i]])
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
