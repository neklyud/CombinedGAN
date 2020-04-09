# -*- coding: utf-8 -*-

from __future__ import division
import os
import time
import random
import glob
import tensorflow.compat.v1 as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size,
                 batch_size, output_size,
                 gf_dim, df_dim, L1_lambda,
                 input_c_dim, output_c_dim, dataset_name,
                 checkpoint_dir, sample_dir, train_size, data_load, 
                 phase,save_latest_freq, print_freq, deviation):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_size = train_size
        self.phase = phase
        self.save_latest_freq = save_latest_freq
        self.print_freq = print_freq
        #self.sample_size = sample_size
        self.output_size = output_size
        self.sample_dir = sample_dir
        self.data_load = data_load
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.deviation = deviation
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')
        self.d_bn6 = batch_norm(name='d_bn6')
        self.d_bn7 = batch_norm(name='d_bn7')
        self.d_bn8 = batch_norm(name='d_bn8')
        self.d_bn9 = batch_norm(name='d_bn9')
        self.d_bn10 = batch_norm(name='d_bn10')
        self.d_bn11 = batch_norm(name='d_bn11')
        self.d_bn12 = batch_norm(name='d_bn12')
        # self.g_bn_e2 = batch_norm(name='g_bn_e2')
        # self.g_bn_e3 = batch_norm(name='g_bn_e3')
        # self.g_bn_e4 = batch_norm(name='g_bn_e4')
        # self.g_bn_e5 = batch_norm(name='g_bn_e5')
        # self.g_bn_e6 = batch_norm(name='g_bn_e6')
        # self.g_bn_e7 = batch_norm(name='g_bn_e7')
        # self.g_bn_e8 = batch_norm(name='g_bn_e8')
        # self.g_bn_e9 = batch_norm(name='g_bn_e9')
        self.g_bn_d0 = batch_norm(name='g_bn_d0')
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d1a = batch_norm(name='g_bn_d1a')
        self.g_bn_d1b = batch_norm(name='g_bn_d1b')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d2a = batch_norm(name='g_bn_d2a')
        self.g_bn_d2b = batch_norm(name='g_bn_d2b')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d3a = batch_norm(name='g_bn_d3a')
        self.g_bn_d3b = batch_norm(name='g_bn_d3b')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d4a = batch_norm(name='g_bn_d4a')
        self.g_bn_d4b = batch_norm(name='g_bn_d4b')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d5a = batch_norm(name='g_bn_d5a')
        self.g_bn_d5b = batch_norm(name='g_bn_d5b')
        self.g_bn_6 = batch_norm(name="g_bn_6")
        self.g_bn_7 = batch_norm(name="g_bn_7")
        self.g_bn_8 = batch_norm(name="g_bn_8")
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):	
        self.real_EVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_c_dim*abs(2*self.deviation - 1)],  name='EVS')
        self.real_SVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_c_dim*abs(2*self.deviation - 1)],  name='SVS')
        self.real_CVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_c_dim*abs(2*self.deviation - 1)], name='real_CVS')
        self.cur_CVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])
        self.cur_SVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])
        self.cur_EVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])

        #############DEVIATE
        #self.dev_EVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, abs(2*self.deviation - 1)*self.input_c_dim], name = 'dev_EVS')
        #self.dev_SVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, abs(2*self.deviation - 1)*self.input_c_dim], name = 'dev_SVS')
        #self.dev_CVS = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, abs(2*self.deviation - 1)*self.input_c_dim], name = 'dev_CVS')
        #############

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        self.real_EVSSVS = tf.concat([self.real_EVS, self.real_SVS], 3)
        self.EVSSVS_view = tf.concat([self.cur_EVS, self.cur_SVS], 2)
        self.cur_EVSSVS = tf.concat([self.cur_EVS, self.cur_SVS], 3)
		# генератор
        self.fake_EN = self.generator(self.real_EVSSVS)
        self.real_ITE = tf.concat([self.cur_EVSSVS, self.cur_CVS], 3)
        self.fake_ITE = tf.concat([self.cur_EVSSVS, self.fake_EN], 3)

        self.EN_view = tf.concat([self.fake_EN, self.cur_CVS], 2)
        print(self.real_ITE.shape)
        print(self.fake_ITE.shape)
        self.D, self.D_logits = self.discriminator(self.real_ITE, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_ITE, reuse=True)

        #self.fake_C_sample = self.sampler(self.real_AB)

        #self.d_sum = tf.summary.histogram("d", self.D)
        #self.d__sum = tf.summary.histogram("d_", self.D_)
        
        self.EVSSVS_view_sum = tf.summary.image("1_SVS_EVS_view", self.EVSSVS_view)
        self.EN_view_sum = tf.summary.image("1_Fusion_cnn_view", self.EN_view)
        self.orig_EVSSVS_view_sum = tf.summary.image("2_TV_IR_view_orig", self.EVSSVS_view)
        self.orig_EN_view_sum = tf.summary.image("2_Fusion_cnn_orig_view", self.EN_view)
        
        self.d_interest = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.g_image = self.L1_lambda * tf.reduce_mean(tf.abs(self.cur_CVS - self.fake_EN))

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.cur_CVS - self.fake_EN))
                        #+ self.L1_lambda * tf.reduce_mean(tf.abs(self.real_C - self.fake_C))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        self.d_interest_loss_sum = tf.summary.scalar("d_interest_loss", self.d_interest)
        self.g_image_loss_sum = tf.summary.scalar("g_image_loss", self.g_image)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=25)

    def load_random_samples_with_deviation(self):
        batch_EVS = []
        batch_SVS = []
        batch_CVS = []
        batch_cur_EVS = []
        batch_cur_SVS = []
        batch_cur_CVS = []
        batch_idxs = min(1e8, self.train_size)//self.batch_size
        rand_idx = 0
        mode = random.randint(0,1)
        if mode == 0:
            rand_idx = random.randint(self.deviation,batch_idxs//3)
        else:
            rand_idx = random.randint((2*batch_idxs)//3, batch_idxs - self.deviation)
        batch_idx = [rand_idx,rand_idx+self.batch_size]
        batch_EVS, batch_SVS, batch_CVS, batch_cur_CVS, batch_cur_SVS, batch_cur_EVS = load_deviation_data(batch_idx,self.data_load,'TRAIN', self.deviation)
        
        sample_image_EVS = np.reshape(batch_EVS, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])
        sample_image_SVS = np.reshape(batch_SVS, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])
        sample_image_CVS = np.reshape(batch_CVS, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])  
        sample_image_cur_EVS = np.reshape(batch_cur_EVS, [self.batch_size, self.image_size, self.image_size, 3])
        sample_image_cur_SVS = np.reshape(batch_cur_SVS, [self.batch_size, self.image_size, self.image_size, 3])
        sample_image_cur_CVS = np.reshape(batch_cur_CVS, [self.batch_size, self.image_size, self.image_size, 3])
        return sample_image_EVS, sample_image_SVS, sample_image_CVS, sample_image_cur_CVS, sample_image_cur_SVS, sample_image_cur_EVS    

    def sample_model_with_deviation(self, sample_dir, epoch, idx, g_optim):
        sample_image_EVS, sample_image_SVS, sample_image_CVS, sample_image_cur_CVS, sample_image_cur_SVS, sample_image_cur_EVS = self.load_random_samples_with_deviation()
        sample_CVS, d_loss, g_loss = self.sess.run(
            [self.fake_EN, self.d_loss, self.g_loss],
            feed_dict = {self.real_EVS: sample_image_EVS, self.real_SVS: sample_image_SVS, self.real_CVS: sample_image_CVS, self.cur_EVS: sample_image_cur_EVS, self.cur_SVS: sample_image_cur_SVS, self.cur_CVS: sample_image_cur_CVS}
        )
        print(sample_image_cur_EVS.shape)
        save_images(sample_image_cur_CVS, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_train_Fusion.png'.format(sample_dir, epoch, idx),self.sample_dir,self.phase)

        save_images(sample_CVS, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_train_Fusion_cnn.png'.format(sample_dir, epoch, idx),self.sample_dir,self.phase)
        print(g_loss)
        if int(idx) % 10 == 0:
            print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

        if g_loss > 500 or g_loss != g_loss:
            #sample_image_EVS, sample_image_SVS, sample_image_CVS, sample_image_cur_CVS, sample_image_cur_SVS, sample_image_cur_EVS = self.load_random_samples_with_deviation()
            # sample_CVS, d_loss, g_loss = self.sess.run(
            #     [self.fake_EN, self.d_loss, self.g_loss],
            #     feed_dict = {self.real_EVS: sample_image_EVS, self.real_SVS: sample_image_SVS, self.real_CVS: sample_image_CVS, self.cur_EVS: sample_image_cur_EVS, self.cur_SVS: sample_image_cur_SVS, self.cur_CVS: sample_image_cur_CVS}
            # )
            # if g_loss > 500 or g_loss != g_loss:
            #     self.load(self.checkpoint_dir)

            g_loss = gradient_clipping(g_optim, g_loss, self.learning_rate, self.beta1)

        return None


    def load_random_samples(self):
        batch_IR = []
        batch_TV = []
        batch_EN = []
        batch_idxs = min(1e8, self.train_size) // self.batch_size 
        #rand_idx = random.randint(0,self.train_size)
        # rand_idx = int(np.random.normal(1000, (1000, 1990), 1)[0])
        # if rand_idx > self.train_size:
        #     rand_idx = self.train_size - 4
        # if rand_idx < 0:
        #     rand_idx = 0
        rand_idx = 0
        mode = random.randint(0,1)
        if mode == 0:
            rand_idx = random.randint(self.deviation,batch_idxs//3)
        else:
            rand_idx = random.randint((2*batch_idxs)//3, batch_idxs - self.deviation)
        batch_idx = [rand_idx,rand_idx+self.batch_size]
        batch_IR_temp, batch_TV_temp, batch_EN_temp = load_data(batch_idx,self.data_load, phase = 'TRAIN')
        sample_image_IR = np.reshape(
            batch_IR_temp, [self.batch_size, self.image_size, self.image_size, 3])
        sample_image_TV = np.reshape(
            batch_TV_temp, [self.batch_size, self.image_size, self.image_size, 3])
        sample_image_EN = np.reshape(
            batch_EN_temp, [self.batch_size, self.image_size, self.image_size, 3])
        
        return sample_image_IR, sample_image_TV, sample_image_EN

    def sample_model(self, sample_dir, epoch, idx):
        sample_image_IR, sample_image_TV, sample_image_EN = self.load_random_samples()
        sample_EN, d_loss, g_loss = self.sess.run(
            [self.fake_EN, self.d_loss, self.g_loss],
            feed_dict={self.real_EVS: sample_image_IR, self.real_SVS: sample_image_TV, self.real_CVS: sample_image_EN})

#        save_images(sample_image_IR, [self.batch_size, 1],
#                    './{}/{:02d}_{:04d}_train_IR.png'.format(sample_dir, epoch, idx),self.sample_dir)
#        save_images(sample_image_TV, [self.batch_size, 1],
#                    './{}/{:02d}_{:04d}_train_TV.png'.format(sample_dir, epoch, idx),self.sample_dir)
        save_images(sample_image_EN, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_train_Fusion.png'.format(sample_dir, epoch, idx),self.sample_dir,self.phase)

        save_images(sample_EN, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}_train_Fusion_cnn.png'.format(sample_dir, epoch, idx),self.sample_dir,self.phase)
        print(g_loss)
        if g_loss > 500 or g_loss != g_loss:
            sample_image_IR, sample_image_TV, sample_image_EN = self.load_random_samples()
            sample_EN, d_loss, g_loss = self.sess.run(
            [self.fake_EN, self.d_loss, self.g_loss],
            feed_dict={self.real_EVS: sample_image_IR, self.real_SVS: sample_image_TV, self.real_CVS: sample_image_EN})
            if g_loss > 500 or g_loss != g_loss:
                self.load(self.checkpoint_dir)
        #if g_loss > 500 and epoch > 2000:
        #    return 'TERMINATE'
        if int(idx) % 10 == 0:
            print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
        return None

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.sum_image = tf.summary.merge([self.EVSSVS_view_sum, self.EN_view_sum])
        self.sum_image_orig = tf.summary.merge([self.orig_EVSSVS_view_sum, self.orig_EN_view_sum])
        self.sum_scalar = tf.summary.merge([self.d_loss_fake_sum, self.g_loss_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_interest_loss_sum, self.g_image_loss_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter_img = self.deviation
        start_time = time.time()
        
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        lr = args.lr
        for epoch in xrange(args.epoch):
            #np.random.shuffle(data)
            batch_idxs = min(1e8, self.train_size - self.deviation) // self.batch_size  
            term = None
            if np.mod(counter_img, 1000) == 1:
                #lr = args.lr * (args.epoch - epoch + 1) / args.epoch
                lr = args.lr * 0.999
            
            for idx in xrange(0, batch_idxs):          
                batch_IR = []
                batch_TV = []
                batch_EN = []
                rand_idx = 0
                mode = random.randint(0,1)
                if mode == 0:
                    rand_idx = random.randint(self.deviation,batch_idxs//3)
                else:
                    rand_idx = random.randint((2*batch_idxs)//3, batch_idxs - self.deviation)
                #rand_idx = random.randint(0,self.train_size)
                batch_idx = [rand_idx*self.batch_size,(rand_idx+1)*self.batch_size]
                batch_IR, batch_TV, batch_EN, batch_cur_CVS, batch_cur_SVS, batch_cur_EVS = load_deviation_data(batch_idx,self.data_load,'TRAIN', self.deviation)
                batch_image_IR = np.reshape(
                    batch_IR, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])
                batch_image_TV = np.reshape(
                    batch_TV, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])
                batch_image_EN = np.reshape(
                    batch_EN, [self.batch_size, self.image_size, self.image_size, 3*(2*self.deviation - 1)])
                batch_cur_CVS_image = np.reshape(batch_cur_CVS, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])
                batch_cur_EVS_image = np.reshape(batch_cur_EVS, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])
                batch_cur_SVS_image = np.reshape(batch_cur_SVS, [self.batch_size, self.image_size, self.image_size, self.output_c_dim])
                _, _, summary_img, summary_str = self.sess.run([d_optim, g_optim, self.sum_image, self.sum_scalar],
                                                                feed_dict={self.real_EVS: batch_image_IR, self.real_SVS: batch_image_TV, self.real_CVS: batch_image_EN, self.cur_CVS: batch_cur_CVS_image, self.cur_SVS: batch_cur_SVS_image, self.cur_EVS: batch_cur_EVS_image, self.learning_rate: lr})

                self.writer.add_summary(summary_str, counter_img)
                if counter_img % self.print_freq == 0:
                    self.writer.add_summary(summary_img, counter_img)

                counter_img += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                    #, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time))#, errD_fake+errD_real, errG))

                if counter_img % self.print_freq == 0:
                    term = self.sample_model_with_deviation(args.sample_dir, epoch, idx, g_optim)
                    print(term)
                    if term != None:
                        break

                if counter_img % self.save_latest_freq == 0:
                    self.save(args.checkpoint_dir, counter_img)
            if term != None:
                break
                    

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 512 x 512 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            print('discriminator')
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            print('h0=', h0.shape)
            # h0 is (256 x 256 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h1_conv')))
            print('h1=', h1.shape)
            # h1 is (128 x 128 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h2_conv')))
            print('h2=', h2.shape)
            # h2 is (64x 64 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*4, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h3_conv')))
            print('h3=', h3.shape)
            # h3 is (32x 32 x self.df_dim*4)
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*8, k_h=4, k_w=4, d_h=2, d_w=2, name='d_h4_conv')))
            print('h4=', h4.shape)
            # h4 is (16 x 16 x self.df_dim*8)
            h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*16, k_h=1, k_w=1, d_h=1, d_w=1, name='d_h5_conv')))
            print('h5=', h5.shape)
            # h4 is (8 x 8 x self.df_dim*8)
            h6 = lrelu(self.d_bn6(conv2d(h5, self.df_dim*16, k_h=1, k_w=1, d_h=1, d_w=1,name='d_h6_conv')))
            print('h6=', h6.shape)
            h7 = lrelu(self.d_bn7(conv2d(h6, self.df_dim*16, k_h=1, k_w=1, d_h=1, d_w=1,name='d_h7_conv')))
            print('h7=', h7.shape)
            h8 = lrelu(self.d_bn8(conv2d(h7, self.df_dim*4, k_h=1, k_w=1, d_h=1, d_w=1,name='d_h8_conv')))
            print('h8=', h8.shape)
            h9 = lrelu(self.d_bn9(conv2d(h8, self.df_dim*2, k_h=3, k_w=3, d_h=1, d_w=1,name='d_h9_conv')))
            print('h9=', h9.shape)
            h10 = lrelu(self.d_bn10(conv2d(h9, self.df_dim*8, k_h=3, k_w=3, d_h=1, d_w=1,name='d_h10_conv')))
            print('h10=', h10.shape)
            h11 = lrelu(self.d_bn11(conv2d(h10, self.df_dim*8,k_h=3, k_w=3, d_h=1, d_w=1, name='d_h11_conv')))
            print('h11=', h11.shape)
            # h4 is (4 x 4 x self.df_dim*8)
            h12 = linear(tf.reshape(h11, [self.batch_size, -1]), 1, 'd_h12_lin')
            print('h12=', h12.shape)
            #return tf.nn.sigmoid(h7), h7
            return tf.nn.sigmoid(h12), h12

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128, s256 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128), int(s/256)
            
            # image is (512 x 512 x self.gf_dim)
            print("image.shape is :", image.shape)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            print("e1.shape is :", e1.shape)
            # e1 is (256 x 256 x self.gf_dim)
            e1a = conv2d(e1, self.gf_dim, name='g_e1a_conv')
            print("e1a.shape is :", e1a.shape)
            # e1a is (128 x 128 x self.gf_dim)
            e2,e2_cache,idx_num = self.conv_block(e1a,3,1,self.gf_dim)
            print("e2.shape is: ", e2.shape)
            print("e2_cache.shape is :", e2_cache.shape)
            # e2_cache is (64 x 64 x self.gf_dim)
            e3,e3_cache,idx_num = self.conv_block(e2_cache,4,idx_num,self.gf_dim*2)
            print("e3_cache.shape is :", e3_cache.shape)
            # e3_cache is (32 x 32 x self.gf_dim*2)
            e4,e4_cache,idx_num = self.conv_block(e3_cache,6,idx_num,self.gf_dim*4)
            print("e4_cache.shape is :", e4_cache.shape)
            # e4_cache is (16 x 16 x self.gf_dim*4)
            e5,e5_cache,idx_num = self.conv_block(e4_cache,6,idx_num,self.gf_dim*8)
            print("e5_cache.shape is :", e5_cache.shape)
            # e4_cache is (8 x 8 x self.gf_dim*4)
            e6,e6_cache,idx_num = self.conv_block(e5_cache,3,idx_num,self.gf_dim*4)
            print("e6_cache.shape is :", e6_cache.shape)
            # e5 is (8 x 8 x self.gf_dim*8)
            #layer
            e7, e7_cache, idx_num = self.conv_block(e6_cache, 3, idx_num, self.gf_dim)
            print("e7_cache.shape is :", e7_cache.shape)
            #layer
            d0  = self.g_bn_d0(conv2d(relu(e7), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d0_conv'))
            d0, _, _ = deconv2d(relu(d0),[self.batch_size, s64, s64, self.gf_dim*16], name='gg_d0', with_w=True)
            d0 = conv2d(relu(self.g_bn_d0(d0)),self.gf_dim*16, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d0a_conv')
            d0 = tf.nn.dropout(relu(self.g_bn_d0(d0)), 0.5)
            d0 = tf.concat([d0, e6], 3)
            print("d0.shape is", d0.shape)
            d1 = self.g_bn_d1(conv2d(relu(d0), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d1_conv'))
            d1, _, _ = deconv2d(relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='gg_d1', with_w=True)
            d1 = conv2d(relu(self.g_bn_d1a(d1)), self.gf_dim*8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d1a_conv')
            d1 = tf.nn.dropout(relu(self.g_bn_d1b(d1)), 0.5)
            #print('d1 = ',d1.shape)
            d1 = tf.concat([d1, e5], 3)
            print("d1.shape is :", d1.shape)
            # d1 is (16 x 16 x self.gf_dim*4)
            d2 = self.g_bn_d2(conv2d(relu(d1), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d2_conv'))
            d2, _, _ = deconv2d(relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*4], name='gg_d2', with_w=True)
            d2 = conv2d(relu(self.g_bn_d2a(d2)), self.gf_dim*4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d2a_conv')
            d2 = tf.nn.dropout(relu(self.g_bn_d2b(d2)), 0.5)
            #print('d2 = ',d2.shape)
            d2 = tf.concat([d2, e4], 3)
            print("d2.shape is :", d2.shape)
            # d2 is (32 x 32 x self.gf_dim*2)

            d3 = self.g_bn_d3(conv2d(relu(d2), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d3_conv'))
            d3, _, _ = deconv2d(relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*2], name='gg_d3', with_w=True)
            d3 = conv2d(relu(self.g_bn_d3a(d3)), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d3a_conv')
            d3 = tf.nn.dropout(relu(self.g_bn_d3b(d3)), 0.5)
            d3 = tf.concat([d3, e3], 3)
            print("d3.shape is :", d3.shape)
            # d3 is (64 x 64 x self.gf_dim*2)

            d4 = self.g_bn_d4(conv2d(relu(d3), self.gf_dim/2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d4_conv'))
            d4, _, _ = deconv2d(relu(d4),
                [self.batch_size, s4, s4, self.gf_dim], name='gg_d4', with_w=True)
            d4 = conv2d(relu(self.g_bn_d4a(d4)), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d4a_conv')
            d4 = tf.nn.dropout(relu(self.g_bn_d4b(d4)), 0.5)
            d4 = tf.concat([d4, e2], 3)
            print("d4.shape is :", d4.shape)
            # d4 is (128 x 128 x self.gf_dim)

            d5 = self.g_bn_d5(conv2d(relu(d4), self.gf_dim/4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d5_conv'))
            d5, _, _ = deconv2d(relu(d5),
                [self.batch_size, s2, s2, self.gf_dim], name='gg_d5', with_w=True)
            d5 = conv2d(relu(self.g_bn_d5a(d5)), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d5a_conv')
            d5 = tf.nn.dropout(relu(self.g_bn_d5b(d5)), 0.5)
            print("old d5.shape: ", d5.shape)
            d5 = tf.concat([d5, e1], 3)
            print("d5.shape is :", d5.shape)
            # d4 is (256 x 256 x self.gf_dim)

            
            d6 = self.g_bn_6(conv2d(relu(d5), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d6a_deconv'))
            print('d6=',d6.shape)
            d6, _, _ = deconv2d(relu(d6),[self.batch_size, s, s, int(self.gf_dim)//2], name='gg_d6', with_w=True)
            print("d6.shape is :", d6.shape)
            # d6 is (512 x 512 x self.gf_dim/2)
            self.d7 = conv2d(relu(self.g_bn_7(d6)), self.gf_dim//2, k_h=3, k_w=3, d_h=1, d_w=1, name='g_d7_conv')
            print("d7.shape is :", self.d7.shape)
            #print('d7 = ',self.d7.shape)
            # d6 is (512 x 512 x self.gf_dim/2)

            self.d8 = conv2d(relu(self.d7), self.output_c_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='g_d8_conv')
            print("d8.shape is :", self.d8.shape)
            # d6 is (512 x 512 x self.output_c_dim)
            return tf.nn.tanh(self.d8)

    #def generator1(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128, s256 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128), int(s/256)
            
            # image is (512 x 512 x self.gf_dim)
            print("image.shape is :", image.shape)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            print("e1.shape is :", e1.shape)
            # e1 is (256 x 256 x self.gf_dim)
            e1a = conv2d(e1, self.gf_dim, name='g_e1a_conv')
            print("e1a.shape is :", e1a.shape)
            # e1a is (128 x 128 x self.gf_dim)
            e2,e2_cache,idx_num = self.conv_block(e1a,3,1,self.gf_dim)
            print("e2_cache.shape is :", e2_cache.shape)
            # e2_cache is (64 x 64 x self.gf_dim)
            e3,e3_cache,idx_num = self.conv_block(e2_cache,4,idx_num,self.gf_dim*2)
            print("e3_cache.shape is :", e3_cache.shape)
            # e3_cache is (32 x 32 x self.gf_dim*2)
            e4,e4_cache,idx_num = self.conv_block(e3_cache,6,idx_num,self.gf_dim*4)
            print("e4_cache.shape is :", e4_cache.shape)
            # e4_cache is (16 x 16 x self.gf_dim*4)
            e5,e5_cache,idx_num = self.conv_block(e4_cache,6,idx_num,self.gf_dim*8)
            print("e5_cache.shape is :", e5_cache.shape)
            # e4_cache is (8 x 8 x self.gf_dim*4)
            e6,e6_cache,idx_num = self.conv_block(e5_cache,3,idx_num,self.gf_dim*4)
            print("e6_cache.shape is :", e6_cache.shape)
            # e5 is (8 x 8 x self.gf_dim*8)
            #layer
            #e7, e7_cache, idx_num = self.conv_block(e6_cache, 3, idx_num, self.gf_dim)
            #print("e7_cache.shape is :". e7_cache.shape)
            #layer
            d1 = self.g_bn_d1(conv2d(relu(e6), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d1_conv'))
            d1, _, _ = deconv2d(relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='gg_d1', with_w=True)
            d1 = conv2d(relu(self.g_bn_d1a(d1)), self.gf_dim*8, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d1a_conv')
            d1 = tf.nn.dropout(relu(self.g_bn_d1b(d1)), 0.5)
            #print('d1 = ',d1.shape)
            d1 = tf.concat([d1, e5], 3)
            print("d1.shape is :", d1.shape)
            # d1 is (16 x 16 x self.gf_dim*4)
            d2 = self.g_bn_d2(conv2d(relu(d1), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d2_conv'))
            d2, _, _ = deconv2d(relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*4], name='gg_d2', with_w=True)
            d2 = conv2d(relu(self.g_bn_d2a(d2)), self.gf_dim*4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d2a_conv')
            d2 = tf.nn.dropout(relu(self.g_bn_d2b(d2)), 0.5)
            #print('d2 = ',d2.shape)
            d2 = tf.concat([d2, e4], 3)
            print("d2.shape is :", d2.shape)
            # d2 is (32 x 32 x self.gf_dim*2)

            d3 = self.g_bn_d3(conv2d(relu(d2), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d3_conv'))
            d3, _, _ = deconv2d(relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*2], name='gg_d3', with_w=True)
            d3 = conv2d(relu(self.g_bn_d3a(d3)), self.gf_dim*2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d3a_conv')
            d3 = tf.nn.dropout(relu(self.g_bn_d3b(d3)), 0.5)
            d3 = tf.concat([d3, e3], 3)
            print("d3.shape is :", d3.shape)
            # d3 is (64 x 64 x self.gf_dim*2)

            d4 = self.g_bn_d4(conv2d(relu(d3), self.gf_dim/2, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d4_conv'))
            d4, _, _ = deconv2d(relu(d4),
                [self.batch_size, s4, s4, self.gf_dim], name='gg_d4', with_w=True)
            d4 = conv2d(relu(self.g_bn_d4a(d4)), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d4a_conv')
            d4 = tf.nn.dropout(relu(self.g_bn_d4b(d4)), 0.5)
            d4 = tf.concat([d4, e2], 3)
            print("d4.shape is :", d4.shape)
            # d4 is (128 x 128 x self.gf_dim)

            d5 = self.g_bn_d5(conv2d(relu(d4), self.gf_dim/4, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d5_conv'))
            d5, _, _ = deconv2d(relu(d5),
                [self.batch_size, s2, s2, self.gf_dim], name='gg_d5', with_w=True)
            d5 = conv2d(relu(self.g_bn_d5a(d5)), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d5a_conv')
            d5 = tf.nn.dropout(relu(self.g_bn_d5b(d5)), 0.5)
            d5 = tf.concat([d5, e1], 3)
            print("d5.shape is :", d5.shape)
            # d4 is (256 x 256 x self.gf_dim)

            
            d6 = self.g_bn_6(conv2d(relu(d5), self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='g_d6a_deconv'))
            #print('d6=',d6.shape)
            d6, _, _ = deconv2d(relu(d6),[self.batch_size, s, s, int(self.gf_dim)//2], name='gg_d6', with_w=True)
            print("d6.shape is :", d6.shape)
            # d6 is (512 x 512 x self.gf_dim/2)
            self.d7 = conv2d(relu(self.g_bn_7(d6)), self.gf_dim//2, k_h=3, k_w=3, d_h=1, d_w=1, name='g_d7_conv')
            print("d7.shape is :", self.d7.shape)
            #print('d7 = ',self.d7.shape)
            # d6 is (512 x 512 x self.gf_dim/2)

            self.d8 = conv2d(relu(self.d7), self.output_c_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='g_d8_conv')
            print("d8.shape is :", self.d8.shape)
            # d6 is (512 x 512 x self.output_c_dim)
            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_dir_res = os.path.join(checkpoint_dir, '%d'%step)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(checkpoint_dir_res):
            os.makedirs(checkpoint_dir_res)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir_res, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        #checkpoint_dir = os.path.join(checkpoint_dir, '5600')
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest != None:
            self.saver.restore(self.sess, latest)
            print("checkpoint was read")
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        data = glob.glob('./datasets/combined_2/Images/TEST/CVS/*.bmp')
        num_files = len(data)
        for idx in xrange(self.deviation, num_files - self.deviation):
            batch_A = []
            batch_B = []
            batch_C = []
            name = str(idx)
            #rnd_num = int(np.random.rand()*num_files)
            rnd_num = idx 
            batch_IR = []
            batch_TV = []
            batch_EN = []

            batch_idx = [rnd_num*self.batch_size,(rnd_num+1)*self.batch_size]
            batch_EVS, batch_SVS, batch_CVS, batch_cur_CVS, batch_cur_SVS, batch_cur_EVS = load_deviation_data(batch_idx,self.data_load, phase = 'TEST', deviation=self.deviation)

            batch_image_EVS = np.reshape(batch_EVS, [self.batch_size, self.image_size, self.image_size, 3*abs(2*self.deviation - 1)])
            batch_image_SVS = np.reshape(batch_SVS, [self.batch_size, self.image_size, self.image_size, 3*abs(2*self.deviation - 1)])
            batch_image_CVS = np.reshape(batch_CVS, [self.batch_size, self.image_size, self.image_size, 3*abs(2*self.deviation - 1)])
            batch_image_cur_EVS = np.reshape(batch_cur_EVS, [self.batch_size, self.image_size, self.image_size, 3])
            batch_image_cur_SVS = np.reshape(batch_cur_SVS, [self.batch_size, self.image_size, self.image_size, 3])
            batch_image_cur_CVS = np.reshape(batch_cur_CVS, [self.batch_size, self.image_size, self.image_size, 3])

            print("Testing image ", idx)
            
            results = self.sess.run(
                    self.fake_EN,
                    feed_dict={self.real_EVS: batch_image_EVS, self.real_SVS: batch_image_SVS, self.real_CVS: batch_image_CVS, self.cur_EVS:batch_image_cur_EVS, self.cur_SVS: batch_image_cur_SVS, self.cur_CVS:batch_image_cur_CVS}
                    )
            #    aaa = (results+1)*127.5
            #   print(np.sum(aaa>254)+np.sum(aaa<1))
            #   if np.sum(aaa>254)+np.sum(aaa<1) > 1000:
            #       i = True
            #   else :
            #       i = False
            
            #save_images(batch_image_IR, [self.batch_size, 1],
            #            './{}/{}_IR.png'.format(args.test_dir, name),000,self.phase)
            #save_images(batch_image_TV, [self.batch_size, 1],
            #            './{}/{}_TV.png'.format(args.test_dir, name),000,self.phase)
            #save_images(batch_image_EN, [self.batch_size, 1],
            #            './{}/{}_Fusion_real.png'.format(args.test_dir, name),000,self.phase)
            # save_images(results, [self.batch_size, 1],
            #             '{}/{}_Fusion_fake.png'.format(args.test_dir, name),000,self.phase)
            print(name)
            print('{}/1_{}.png'.format(args.test_dir, name))
            save_images(results, [self.batch_size, 1],
                        '{}/_1_{}.png'.format(args.test_dir, name),000,self.phase)
            save_images(batch_image_cur_EVS, [self.batch_size, 1],
                        '{}/_2_{}'.format(args.test_dir, name),000,self.phase)
            save_images(batch_image_cur_SVS, [self.batch_size, 1],
                        '{}/_3_{}.png'.format(args.test_dir, name),000,self.phase)
            save_images(batch_image_cur_CVS, [self.batch_size, 1],
                       '{}/_4_{}.png'.format(args.test_dir, name),000,self.phase)
    
    def conv_block(self, inputs,rep,idx,dim):
        e2 = inputs
        for i in range(rep-1):
            ec = e2
            idx+=1
            e1 = conv2d(lrelu(ec), dim,k_h=3, k_w=3, d_h=1, d_w=1, name='g_e%d_conv'%idx)
            idx+=1
            e2 = conv2d(lrelu(e1), dim,k_h=3, k_w=3, d_h=1, d_w=1, name='g_e%d_conv'%idx)
            e2 = tf.concat([e2, ec], 3)

        ec = e2
        idx+=1
        e1 = conv2d(lrelu(ec), dim,k_h=3, k_w=3, d_h=1, d_w=1, name='g_e%d_conv'%idx)
        idx+=1
        e2 = conv2d(lrelu(e1), dim,k_h=3, k_w=3, d_h=1, d_w=1, name='g_e%d_conv'%idx)
        e2 = tf.concat([e2, ec], 3)
        e2_cache = conv2d(lrelu(e1), dim,k_h=3, k_w=3, name='g_e%d_cache_conv'%idx)
        e2_cache = tf.concat([e2_cache, conv2d(ec, dim, k_h=1, k_w=1, name='g_e%dc_cache_conv'%idx)], 3)
        return e2,e2_cache,idx
