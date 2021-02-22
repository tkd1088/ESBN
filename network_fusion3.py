import inspect
import os
import numpy as np
import tensorflow as tf


class MCAE_fusion:
    def __init__(self, input_LDR_low, input_LDR_mid, input_LDR_high, input_HDR_low, input_HDR_mid, input_HDR_high, is_train):    
        scope = "fusion"

        low = tf.concat([input_LDR_low, input_HDR_low],3)
        mid = tf.concat([input_LDR_mid, input_HDR_mid],3)
        high = tf.concat([input_LDR_high, input_HDR_high],3)

        low = self.conv_layer(low, scope, name="conv1_low", kshape=[3,3,6,64], strides=[1,1,1,1])
        low = self.lrelu(low, scope, "lrelu1_low")
        mid = self.conv_layer(mid, scope, name="conv1_mid", kshape=[3,3,6,64], strides=[1,1,1,1])
        mid = self.lrelu(mid, scope, "lrelu1_mid")
        high = self.conv_layer(high, scope, name="conv1_high", kshape=[3,3,6,64], strides=[1,1,1,1])
        high = self.lrelu(high, scope, "lrelu1_high")

        x_0 = tf.concat([low, mid, high],3) #64*3
        x_0 = self.conv_layer(x_0, scope, name="conv2", kshape=[3,3,64*3,64], strides=[1,1,1,1])
        x_0 = self.lrelu(x_0, scope, "lrelu2")

        x_1 = self.resden_block(x_0, is_train, scope, "resden1") #64

        #x_2 = self.resden_block(x_1, is_train, scope, "resden2") #64

        #x_3 = self.resden_block(x_2, is_train, scope, "resden3") #64

        #x_4 = tf.concat([x_1, x_2],3) #64*2

        #x_5 = self.conv_layer(x_4, scope, name="conv3", kshape=[3,3,64*2,64], strides=[1,1,1,1])
        #x_5 = self.lrelu(x_5, scope, "lrelu3")

        x_6 = x_1 + mid

        x_7 = self.conv_layer(x_6, scope, name="conv4", kshape=[3,3,64,3], strides=[1,1,1,1])
        x_7 = tf.nn.tanh(x_7)

        self.output_hdr = x_7

    def resden_block(self, bottom, is_train, scope, name):
        y_0 = bottom

        y_1 = y_0 # 64
        y_1 = self.conv_layer(y_0, scope, name=name+"_conv1", kshape=[3,3,64+32*0,32], strides=[1,1,1,1])
        y_1 = self.lrelu(y_1, scope, name+"_lrelu1")

        y_2 = tf.concat([y_0, y_1],3) #64+32*1
        y_2 = self.conv_layer(y_2, scope, name=name+"_conv2", kshape=[3,3,64+32*1,32], strides=[1,1,1,1])
        y_2 = self.lrelu(y_2, scope, name+"_lrelu2")

        y_3 = tf.concat([y_0, y_1, y_2],3) #64+32*2
        y_3 = self.conv_layer(y_3, scope, name=name+"_conv3", kshape=[3,3,64+32*2,32], strides=[1,1,1,1])
        y_3 = self.lrelu(y_3, scope, name+"_lrelu3")

        y_4 = tf.concat([y_0, y_1, y_2, y_3],3) #64+32*3
        y_4 = self.conv_layer(y_4, scope, name=name+"_conv4", kshape=[3,3,64+32*3,32], strides=[1,1,1,1])
        y_4 = self.lrelu(y_4, scope, name+"_lrelu4")

        y_5 = tf.concat([y_0, y_1, y_2, y_3, y_4],3) #64+32*4
        y_5 = self.conv_layer(y_5, scope, name=name+"_conv5", kshape=[3,3,64+32*4,32], strides=[1,1,1,1])
        y_5 = self.lrelu(y_5, scope, name+"_lrelu5")

        y_6 = tf.concat([y_0, y_1, y_2, y_3, y_4, y_5],3) #64+32*5
        y_6 = self.conv_layer(y_6, scope, name=name+"_conv6", kshape=[3,3,64+32*5,32], strides=[1,1,1,1])
        y_6 = self.lrelu(y_6, scope, name+"_lrelu6")

        y_7 = tf.concat([y_0, y_1, y_2, y_3, y_4, y_5, y_6],3) #64+32*6
        y_7 = self.conv_layer(y_7, scope, name=name+"_conv7", kshape=[1,1,64+32*6,64], strides=[1,1,1,1])
        y_7 = self.lrelu(y_7, scope, name+"_lrelu7")

        return y_0 + y_7



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


    def lrelu(self, bottom, scope, name, leak=0.2):
        with tf.variable_scope(scope+"_"+name):
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




