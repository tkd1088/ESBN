import network_align2
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import time
import math


# In[2]: parameter setting 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_HEIGHT = 224*4
IMG_WIDTH = 224*6
IMG_DEPTH = 3

is_training = False
gamma = 2.2

test_path = "/data/tkd1088/dataset/Kalantari/Test/EXTRA"


# In[7]: Network design
input_LDR_low = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
input_LDR_mid = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
input_LDR_high = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
input_HDR_low = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
input_HDR_mid = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
input_HDR_high = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gt_LDR_low= tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gt_LDR_high = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gt_HDR = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
is_train = tf.placeholder(tf.bool, name='is_train')




# In[7]: Network design
my_CAE = network_align2.MCAE_align(input_LDR_low, input_LDR_mid, input_HDR_low, input_HDR_mid, is_train)
output_low = my_CAE.output_low
#output_high = my_CAE.output_high


# In[8]: training session
nscene = 15


saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    min_iteration = 220000
    max_iteration = 220000

    for iteration in range(min_iteration, max_iteration+1, 10000) :
        ckpt_low_path = "/data/tkd1088/result/hdr_transfer/ckpt_190417_1_low/iter_most"
        ckpt_high_path = "/data/tkd1088/result/hdr_transfer/ckpt_190417_1_high/iter_most"

        result_path = "/data/tkd1088/dataset/Kalantari/Test/OURS"
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        saver.restore(sess, ckpt_low_path)
        total_psnr = 0
        for n in range(1, nscene+1):

            scene_path = '%s/%03d' % (test_path,n)
            image_pathss = '%s/*.tif' % scene_path
            image_paths = sorted(glob.glob(image_pathss))
            info_path = '%s/exposure.txt' % scene_path
            hdr_path = '%s/HDRImg.hdr' % scene_path

            input_LDR_low_ = cv2.imread(image_paths[0]).astype(np.float32)/255.
            input_LDR_mid_ = cv2.imread(image_paths[1]).astype(np.float32)/255.
            gt_HDR_ = cv2.imread(hdr_path,-1).astype(np.float32)

            input_LDR_low_ = input_LDR_low_[:,:,::-1]
            input_LDR_mid_ = input_LDR_mid_[:,:,::-1]
            gt_HDR_ = gt_HDR_[:,:,::-1]

            input_LDR_low_ = cv2.resize(input_LDR_low_, (IMG_WIDTH,IMG_HEIGHT))
            input_LDR_mid_ = cv2.resize(input_LDR_mid_, (IMG_WIDTH,IMG_HEIGHT))
            gt_HDR_ = cv2.resize(gt_HDR_, (IMG_WIDTH,IMG_HEIGHT))

            height, width, channel = input_LDR_mid_.shape

            expo = np.zeros(3)
            file = open(info_path, 'r')
            expo[0]= np.power(2.0, float(file.readline()))
            expo[1]= np.power(2.0, float(file.readline()))
            expo[2]= np.power(2.0, float(file.readline()))
            file.close()

            input_HDR_low_ = np.power(input_LDR_low_, gamma)/expo[0]
            input_HDR_mid_ = np.power(input_LDR_mid_, gamma)/expo[1]

            gt_LDR_low_ = np.clip(gt_HDR_*expo[0], 0.0, 1.0)
            gt_LDR_mid_ = np.clip(gt_HDR_*expo[1], 0.0, 1.0)
            gt_LDR_low_ = np.power(gt_LDR_low_, 1./gamma)
            gt_LDR_mid_ = np.power(gt_LDR_mid_, 1./gamma)
            
            gt_HDR_low_ = np.power(gt_LDR_low_, gamma)/expo[0]
            gt_HDR_mid_ = np.power(gt_LDR_mid_, gamma)/expo[1]

            input_LDR_low_ = (255.*input_LDR_low_).astype(np.uint8)
            input_LDR_mid_ = (255.*input_LDR_mid_).astype(np.uint8)

            gt_LDR_low_ = (255.*gt_LDR_low_).astype(np.uint8)
            gt_LDR_mid_ = (255.*gt_LDR_mid_).astype(np.uint8)

            #cv2.imwrite('%s/%03d_ours2_aligned_1_gt.png' % (result_path, n), gt_LDR_low_[:,:,::-1])
            #cv2.imwrite('%s/%03d_ours2_aligned_2_gt.png' % (result_path, n), gt_LDR_mid_[:,:,::-1])
            cv2.imwrite('%s/%03d/ours_aligned_2.png' % (result_path, n), input_LDR_mid_[:,:,::-1])
            #cv2.imwrite('%s/%03d_recon_0.png' % (result_path, n), gt_LDR_low_[:,:,::-1])
            #cv2.imwrite('%s/%03d_recon_2.png' % (result_path, n), gt_LDR_mid_[:,:,::-1])

        
            batch_input_LDR_low = []
            batch_input_LDR_mid = []
            batch_input_HDR_low = []
            batch_input_HDR_mid = []
        
            input_LDR_low_ = input_LDR_low_.astype(np.float32)/127.5 -1.
            input_LDR_mid_ = input_LDR_mid_.astype(np.float32)/127.5 -1.
            #input_LDR_mid_ = gt_LDR_mid_.astype(np.float32)/127.5 -1.
            input_HDR_low_ = input_HDR_low_ *2. -1.
            input_HDR_mid_ = input_HDR_mid_ *2. -1.

            batch_input_LDR_low.append(input_LDR_low_)
            batch_input_LDR_mid.append(input_LDR_mid_)
            batch_input_HDR_low.append(input_HDR_low_)
            batch_input_HDR_mid.append(input_HDR_mid_)

            batch_input_LDR_low_tensor = np.stack(batch_input_LDR_low, axis=0)
            batch_input_LDR_mid_tensor = np.stack(batch_input_LDR_mid, axis=0)
            batch_input_HDR_low_tensor = np.stack(batch_input_HDR_low, axis=0)
            batch_input_HDR_mid_tensor = np.stack(batch_input_HDR_mid, axis=0)

            st = time.time()         
            recon_low = sess.run(output_low, feed_dict={
                                  input_LDR_low    : batch_input_LDR_low_tensor,
                                  input_LDR_mid    : batch_input_LDR_mid_tensor,
                                  input_HDR_low    : batch_input_HDR_low_tensor,
                                  input_HDR_mid    : batch_input_HDR_mid_tensor,
                                  is_train         : is_training})
            ed = time.time() 
            
            img_recon_low_ = recon_low[0,:,:,::-1].astype(np.float32)
            img_recon_low_ = np.clip(img_recon_low_, -1.0, 1.0)
            img_recon_low_ += 1.
            img_recon_low_ *= 127.5

            mse = np.mean((gt_LDR_low_[:,:,::-1]/255.-img_recon_low_/255.)**2)
            psnr = -10.*math.log10(mse)
            total_psnr += psnr
        
        
            cv2.imwrite('%s/%03d/ours_aligned_1.png' % (result_path, n), img_recon_low_)
            print('Iteration %06d Scene %03d low is reconstructed (%.4f)' % (iteration,n,ed-st))


        saver.restore(sess, ckpt_high_path)

        for n in range(1, nscene+1):

            scene_path = '%s/%03d' % (test_path,n)
            image_pathss = '%s/*.tif' % scene_path
            image_paths = sorted(glob.glob(image_pathss))
            info_path = '%s/exposure.txt' % scene_path
            hdr_path = '%s/HDRImg.hdr' % scene_path

            input_LDR_mid_ = cv2.imread(image_paths[1]).astype(np.float32)/255.
            input_LDR_high_ = cv2.imread(image_paths[2]).astype(np.float32)/255.
            gt_HDR_ = cv2.imread(hdr_path,-1).astype(np.float32)

            input_LDR_mid_ = input_LDR_mid_[:,:,::-1]
            input_LDR_high_ = input_LDR_high_[:,:,::-1]
            gt_HDR_ = gt_HDR_[:,:,::-1]

            input_LDR_mid_ = cv2.resize(input_LDR_mid_, (IMG_WIDTH,IMG_HEIGHT))
            input_LDR_high_ = cv2.resize(input_LDR_high_, (IMG_WIDTH,IMG_HEIGHT))
            gt_HDR_ = cv2.resize(gt_HDR_, (IMG_WIDTH,IMG_HEIGHT))

            height, width, channel = input_LDR_mid_.shape

            expo = np.zeros(3)
            file = open(info_path, 'r')
            expo[0]= np.power(2.0, float(file.readline()))
            expo[1]= np.power(2.0, float(file.readline()))
            expo[2]= np.power(2.0, float(file.readline()))
            file.close()

            input_HDR_mid_ = np.power(input_LDR_mid_, gamma)/expo[1]
            input_HDR_high_ = np.power(input_LDR_high_, gamma)/expo[2]

            gt_LDR_mid_ = np.clip(gt_HDR_*expo[1], 0.0, 1.0)
            gt_LDR_high_ = np.clip(gt_HDR_*expo[2], 0.0, 1.0)
            gt_LDR_mid_ = np.power(gt_LDR_mid_, 1./gamma)
            gt_LDR_high_ = np.power(gt_LDR_high_, 1./gamma)
            
            gt_HDR_mid_ = np.power(gt_LDR_mid_, gamma)/expo[1]
            gt_HDR_high_ = np.power(gt_LDR_high_, gamma)/expo[2]

            input_LDR_mid_ = (255.*input_LDR_mid_).astype(np.uint8)
            input_LDR_high_ = (255.*input_LDR_high_).astype(np.uint8)

            gt_LDR_mid_ = (255.*gt_LDR_mid_).astype(np.uint8)
            gt_LDR_high_ = (255.*gt_LDR_high_).astype(np.uint8)

            #cv2.imwrite('%s/%03d_ours2_aligned_3_gt.png' % (result_path, n), gt_LDR_high_[:,:,::-1])

        
            batch_input_LDR_mid = []
            batch_input_LDR_high = []
            batch_input_HDR_mid = []
            batch_input_HDR_high = []
        
            input_LDR_mid_ = gt_LDR_mid_.astype(np.float32)/127.5 -1.
            input_LDR_high_ = input_LDR_high_.astype(np.float32)/127.5 -1.
            input_HDR_mid_ = input_HDR_mid_ *2. -1.
            input_HDR_high_ = input_HDR_high_ *2. -1.

            batch_input_LDR_mid.append(input_LDR_mid_)
            batch_input_LDR_high.append(input_LDR_high_)
            batch_input_HDR_mid.append(input_HDR_mid_)
            batch_input_HDR_high.append(input_HDR_high_)

            batch_input_LDR_mid_tensor = np.stack(batch_input_LDR_mid, axis=0)
            batch_input_LDR_high_tensor = np.stack(batch_input_LDR_high, axis=0)
            batch_input_HDR_mid_tensor = np.stack(batch_input_HDR_mid, axis=0)
            batch_input_HDR_high_tensor = np.stack(batch_input_HDR_high, axis=0)


            st = time.time()        
            recon_low = sess.run(output_low, feed_dict={
                                  input_LDR_low    : batch_input_LDR_high_tensor,
                                  input_LDR_mid    : batch_input_LDR_mid_tensor,
                                  input_HDR_low    : batch_input_HDR_high_tensor,
                                  input_HDR_mid    : batch_input_HDR_mid_tensor,
                                  is_train         : is_training})
            ed = time.time()
            
            img_recon_low_ = recon_low[0,:,:,::-1].astype(np.float32)
            img_recon_low_ = np.clip(img_recon_low_, -1.0, 1.0)
            img_recon_low_ += 1.
            img_recon_low_ *= 127.5

            mse = np.mean((gt_LDR_high_[:,:,::-1]/255.-img_recon_low_/255.)**2)
            psnr = -10.*math.log10(mse)
            total_psnr += psnr
        
            cv2.imwrite('%s/%03d/ours_aligned_3.png' % (result_path, n), img_recon_low_)
            print('Iteration %06d Scene %03d high is reconstructed (%.4f)' % (iteration,n,ed-st))
        print('Iteration %06d is reconstructed with psnr %f' % (iteration, total_psnr/nscene/2))
