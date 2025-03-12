''' @author: Wu
'''
import tensorflow as tf
import numpy as np
from utils import *

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

class SR_NETWORK(object):
    def __init__(self, x_LR=None, x_HR=None, r=None, status='pretraining', alpha_advers=0.001):

        status = status.lower()
        if status not in ['pretraining', 'training', 'testing']:
            print('Error in network status.')
            exit()

        self.x_LR, self.x_HR = x_LR, x_HR

        if r is None:
            print('Error in SR scaling. Variable r must be specified.')
            exit()

        if status in ['pretraining', 'training']:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=True)
        else:
            self.x_SR = self.generator(self.x_LR, r=r, is_training=False)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        if status == 'pretraining':
            #self.g_loss = self.compute_losses(self.x_HR, self.x_SR, None, None, alpha_advers, isGAN=False)
            g_loss_out = self.compute_losses(self.x_HR, self.x_SR, None, None, alpha_advers, isGAN=False)
            self.g_loss = g_loss_out[0]
            self.ratio = g_loss_out[1]            

            self.d_loss, self.disc_HR, self.disc_SR, self.d_variables = None, None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None

        elif status == 'training':
            self.disc_HR = self.discriminator(self.x_HR, reuse=False)
            self.disc_SR = self.discriminator(self.x_SR, reuse=True)
            self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            loss_out = self.compute_losses(self.x_HR, self.x_SR, self.disc_HR, self.disc_SR, alpha_advers, isGAN=True)
            self.g_loss = loss_out[0]
            self.d_loss = loss_out[1]
            self.advers_perf = loss_out[2]
            self.content_loss = loss_out[3]
            self.g_advers_loss  = loss_out[4]

        else:
            self.g_loss, self.d_loss = None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None
            self.advers_perf, self.content_loss, self.g_advers_loss = None, None, None
            self.disc_HR, self.disc_SR, self.d_variables = None, None, None


    def generator(self, x, r, is_training=False, reuse=False):
        if is_training:
            N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]
        else:
            N, h, w, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.get_shape()[3]

        k, stride = 3, 1
        output_shape = [N, h+2*k, w+2*k, -1]

        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                C_in, C_out = C, 64
                output_shape[-1] = C_out
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.nn.relu(x)

            skip_connection = x

            # B residual blocks
            C_in, C_out = C_out, 64
            output_shape[-1] = C_out
            for i in range(16):
                B_skip_connection = x

                with tf.variable_scope('block_{}a'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.nn.relu(x)

                with tf.variable_scope('block_{}b'.format(i+1)):
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)

                x = tf.add(x, B_skip_connection)

            with tf.variable_scope('deconv2'):
                x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                x = tf.add(x, skip_connection)

            # Super resolution scaling
            r_prod = 1
            for i, r_i in enumerate(r):
                C_out = (r_i**2)*C_in
                with tf.variable_scope('deconv{}'.format(i+3)):
                    output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C_out]
                    x = deconv_layer_2d(x, [k, k, C_out, C_in], output_shape, stride, k)
                    x = tf.depth_to_space(x, r_i)
                    x = tf.nn.relu(x)

                r_prod *= r_i

            output_shape = [N, r_prod*h+2*k, r_prod*w+2*k, C]
            with tf.variable_scope('deconv_out'):
                x = deconv_layer_2d(x, [k, k, C, C_in], output_shape, stride, k)

        return x


    def discriminator(self, x, reuse=False):
        N, h, w, C = tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], x.get_shape()[3]

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer_2d(x, [3, 3, C, 32], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv2'):
                x = conv_layer_2d(x, [3, 3, 32, 32], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv3'):
                x = conv_layer_2d(x, [3, 3, 32, 64], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv4'):
                x = conv_layer_2d(x, [3, 3, 64, 64], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv5'):
                x = conv_layer_2d(x, [3, 3, 64, 128], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv6'):
                x = conv_layer_2d(x, [3, 3, 128, 128], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv7'):
                x = conv_layer_2d(x, [3, 3, 128, 256], 1)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('conv8'):
                x = conv_layer_2d(x, [3, 3, 256, 256], 2)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            x = flatten_layer(x)
            with tf.variable_scope('fully_connected1'):
                x = dense_layer(x, 1024)
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('fully_connected2'):
                x = dense_layer(x, 1)

        return x

    def tf_diffvy(self, x_SR):
        v_SR_gradienty = x_SR[:,1:,:,1]-x_SR[:,:-1,:,1]
        return v_SR_gradienty[:,0:199,0:249]

    def tf_diffux(self, x_SR):
        u_SR_gradientx = x_SR[:,:,:,0]-x_SR[:,:,:,1]
        return u_SR_gradientx[:,0:199,0:249]

    def tf_diffvz(self, x_SR):
        v_SR_gradientz = x_SR[:,:,1:,1]-x_SR[:,:,:-1,1]
        return v_SR_gradientz[:,0:199,0:249]

    def tf_diffwy(self, x_SR):
        w_SR_gradienty = x_SR[:,1:,:,2]-x_SR[:,:-1,:,2]
        return w_SR_gradienty[:,0:199,0:249]

    def tf_diffwz(self, x_SR):
        w_SR_gradientz = x_SR[:,:,1:,2]-x_SR[:,:,:-1,2]
        return w_SR_gradientz[:,0:199,0:249]
    
    def compute_losses(self, x_HR, x_SR, d_HR, d_SR, alpha_advers=0.001, isGAN=False):
        
        content_loss = tf.reduce_mean((x_HR - x_SR)**2, axis=[1, 2, 3])
        print(x_HR.shape)

        '''
        v_SR_gradienty = self.tf_diffvy(x_SR)
        u_SR_gradientx = self.tf_diffux(x_SR)
        w_SR_gradientz = self.tf_diffwz(x_SR)        
        
        v_SR_gradientz = self.tf_diffvz(x_SR)
        w_SR_gradienty = self.tf_diffwy(x_SR)
        
        v_HR_gradientz = self.tf_diffvz(x_HR)
        w_HR_gradienty = self.tf_diffwy(x_HR)
        
        v_HR_gradienty = self.tf_diffvy(x_HR)
        w_HR_gradientz = self.tf_diffwz(x_HR)       
        '''
        #conti_loss = tf.reduce_mean((4*v_SR_gradienty[:,:,:] + w_SR_gradientz[:,:,:] + u_SR_gradientx[:,:,:])**2, axis=[1, 2])        

        #g_loss = tf.reduce_mean(content_loss) + 0.2*tf.reduce_mean(conti_loss)
        g_loss = tf.reduce_mean(content_loss)
        ratio = 0.5*tf.reduce_mean(content_loss)/g_loss     


        if isGAN:
            g_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR))

            d_advers_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                                                                    labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0))

            advers_perf = [tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) > 0.5, tf.float32)), # % true positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) < 0.5, tf.float32)), # % true negative
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_SR) > 0.5, tf.float32)), # % false positive
                           tf.reduce_mean(tf.cast(tf.sigmoid(d_HR) < 0.5, tf.float32))] # % false negative

            
            
            g_loss = tf.reduce_mean(content_loss) + alpha_advers*tf.reduce_mean(g_advers_loss)
            #g_loss = tf.reduce_mean(content_loss) + 0.2*tf.reduce_mean(conti_loss) + alpha_advers*tf.reduce_mean(g_advers_loss)
            d_loss = tf.reduce_mean(d_advers_loss)

            return g_loss, d_loss, advers_perf, content_loss, g_advers_loss
        else:
            return g_loss, ratio 
    

        
