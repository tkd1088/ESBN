import inspect
import os
import numpy as np
import tensorflow as tf


class MCAE_align:
    #def __init__(self, input_LDR_low, input_LDR_mid, input_LDR_high, input_HDR_low, input_HDR_mid, input_HDR_high, is_train):
    def __init__(self, input_LDR_low, input_LDR_mid, input_HDR_low, input_HDR_mid, is_train):    
        
        low_2, low_1, low_0 = self.encoder(input_LDR_low, input_HDR_low, is_train, "Align_Encoder_low")
        midl_2, midl_1, midl_0= self.encoder(input_LDR_mid, input_HDR_mid, is_train, "Align_Encoder_midforlow")
        self.output_low = self.decoder_align(low_2, low_1, low_0, midl_2, midl_1, midl_0, is_train, "Align_Decoder_low")
        
        #high_6, high_5, high_4, high_3, high_2, high_1, high_0 = self.encoder(input_LDR_high, input_HDR_high, is_train, "Align_Encoder_high")
        #midh_6, midh_5, midh_4, midh_3, midh_2, midh_1, midh_0 = self.encoder(input_LDR_mid, input_HDR_mid, is_train, "Align_Encoder_midforhigh")
        #self.output_high = self.decoder_align(high_6, high_5, high_4, high_3, high_2, high_1, high_0, midl_6, midl_5, midl_4, midl_3, midl_2, midl_1, midl_0, is_train, "Align_Decoder_high")


    def encoder(self, bottom_ldr, bottom_hdr, is_train, scope) :
        with tf.name_scope(scope) :
            x_0_ldr = bottom_ldr
            x_0_hdr = bottom_hdr
            x_0 = tf.concat([x_0_ldr, x_0_hdr], 3)

            x_1 = self.conv_layer(x_0, scope, "conv1", kshape=[5,5,6,64], strides=[1,2,2,1])
            
            x_2 = self.lrelu(x_1, "lrelu2")
            x_2 = self.conv_layer(x_2, scope, "conv2", kshape=[5,5,64,128], strides=[1,2,2,1])
            x_2 = self.batch_normalization(x_2, is_train, scope, "batn2")

            return x_2, x_1, x_0_ldr


    def decoder_align(self, b1_2, b1_1, b1_0, b2_2, b2_1, b2_0, is_train, scope) :
        with tf.name_scope(scope) : 
            x_4 = tf.concat([b1_2, b2_2], 3) #64*64*(128*2)

            x_3 = self.lrelu(x_4, "lrelu3")
            x_3 = self.conv_layer(x_3, scope, "conv3", kshape=[5,5,256,256], strides=[1,2,2,1])
            x_3 = self.batch_normalization(x_3, is_train, scope, "batn3")

            res = x_3 #32*32*256
            for i in range(9) :
                res = self.res_block(res, is_train, scope, "resb%d"%(i))

            x_2 = tf.concat([res, x_3], 3) # 32*32*(256*2)
            x_2 = tf.nn.relu(x_2)
            x_2 = self.upsample(x_2, scope, "upsa2")
            x_2 = self.conv_layer(x_2, scope, "conv2", kshape=[5,5,512,128], strides=[1,1,1,1])
            x_2 = self.batch_normalization(x_2, is_train, scope, "batn2")

            x_1 = tf.concat([x_2, b1_2, b2_2], 3) # 64*64*(128*3)
            x_1 = tf.nn.relu(x_1)
            x_1 = self.upsample(x_1, scope, "upsa1")
            x_1 = self.conv_layer(x_1, scope, "conv1", kshape=[5,5,384,64], strides=[1,1,1,1])
            x_1 = self.batch_normalization(x_1, is_train, scope, "batn1")

            x_0 = tf.concat([x_1, b1_1, b2_1], 3) # 128*128*(64*3)
            x_0 = tf.nn.relu(x_0)
            x_0 = self.upsample(x_0, scope, "upsa0")
            x_0 = self.conv_layer(x_0, scope, "conv0", kshape=[5,5,192,64], strides=[1,1,1,1])
            x_0 = self.batch_normalization(x_0, is_train, scope, "batn0")

            x = tf.nn.relu(x_0)# 256*256*64
            x = self.conv_layer(x, scope, "conv8", kshape=[5,5,64,3], strides=[1,1,1,1])
            x = tf.nn.tanh(x)

            return x


    def res_block(self, bottom, is_train, scope, name):
        y = self.conv_layer(bottom, scope, name=name+"_conv1", kshape=[3,3,256,256], strides=[1,1,1,1])
        y = self.batch_normalization(y, is_train, scope, name=name+"_batn1")
        y = tf.nn.relu(y)
        y = self.conv_layer(y, scope, name=name+"_conv2", kshape=[3,3,256,256], strides=[1,1,1,1])
        y = self.batch_normalization(y, is_train, scope, name=name+"_batn2")
        return bottom + y

    def avg_pool(self, bottom, scope, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+"_"+name)

    def max_pool(self, bottom, scope, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+"_"+name)

    def upsample(self, bottom, scope, name, factor=[2,2]):
        size = [int(bottom.shape[1] * factor[0]), int(bottom.shape[2] * factor[1])]
        with tf.name_scope(scope+"_"+name):
            return tf.image.resize_nearest_neighbor(bottom, size=size, align_corners=None, name=None)

    def batch_normalization(self, bottom, is_train, scope, name) :
        return tf.contrib.layers.batch_norm(bottom, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_train, scope=scope+"_"+name)


    def lrelu(self, bottom, name, leak=0.2):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * bottom + f2 * abs(bottom)

    def conv_layer(self, bottom, scope, name, kshape, strides=[1, 1, 1, 1]):
        with tf.variable_scope(scope+"_"+name):
            W = tf.get_variable(name=scope+"_"+name+"_weights",
                                shape=kshape,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable(name=scope+"_"+name+"_biases",
                                shape=[kshape[3]],
                                initializer=tf.constant_initializer(0.0))
            out = tf.nn.conv2d(bottom,W,strides=strides, padding='SAME')
            out = tf.nn.bias_add(out, b)

            return out



