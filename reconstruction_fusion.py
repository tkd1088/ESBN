
import network_fusion3
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import math
import time
from skimage.measure import compare_ssim as ssim

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
my_CAE = network_fusion3.MCAE_fusion(input_LDR_low, input_LDR_mid, input_LDR_high, input_HDR_low, input_HDR_mid, input_HDR_high, is_train)
output_hdr = my_CAE.output_hdr


# In[8]: test session
#align_exp = 'recon_190417_2_220000'
align_exp1 = 'recon_190510_1_280000'
align_exp2 = 'recon_190510_1_280000'
nscene = 15

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    min_iteration = 250000
    max_iteration = 250000

    for iteration in range(min_iteration, max_iteration+1, 10000) :
        result_path = "/data/tkd1088/result/hdr_transfer/results_compare/result_191116"
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        ckpt_hdr_path = "/data/tkd1088/result/hdr_transfer/ckpt_190505_1_hdr/iter_%06d" % iteration
        saver.restore(sess, ckpt_hdr_path)

        total_psnrT = 0
        total_psnrL = 0
        total_ssimT = 0
        total_ssimL = 0

        for n in range(1, nscene+1):

          scene_path = '%s/%03d' % (test_path,n)
          image_pathss = '%s/LDR_static_*.png' % scene_path
          image_paths = sorted(glob.glob(image_pathss))
          info_path = '%s/exposure.txt' % scene_path
          hdr_path = '%s/HDRImg.hdr' % scene_path
          recon_low_path = '/data/tkd1088/result/hdr_transfer/results_compare/aligned/%03d_ours2_aligned_1.png' % n
          recon_mid_path = '/data/tkd1088/result/hdr_transfer/results_compare/aligned/%03d_ours2_aligned_2.png' % n 
          recon_high_path = '/data/tkd1088/result/hdr_transfer/results_compare/aligned/%03d_ours2_aligned_3.png' % n

          input_LDR_low_ = cv2.imread(recon_low_path).astype(np.float32)/255.
          input_LDR_mid_ = cv2.imread(recon_mid_path).astype(np.float32)/255.
          input_LDR_high_ = cv2.imread(recon_high_path).astype(np.float32)/255.
          gt_HDR_ = cv2.imread(hdr_path,-1).astype(np.float32)

          input_LDR_low_ = input_LDR_low_[:,:,::-1]
          input_LDR_mid_ = input_LDR_mid_[:,:,::-1]
          input_LDR_high_ = input_LDR_high_[:,:,::-1]
          gt_HDR_ = gt_HDR_[:,:,::-1]

          input_LDR_low_ = cv2.resize(input_LDR_low_, (IMG_WIDTH,IMG_HEIGHT))
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

          input_HDR_low_ = np.power(input_LDR_low_, gamma)/expo[0]
          input_HDR_mid_ = np.power(input_LDR_mid_, gamma)/expo[1]
          input_HDR_high_ = np.power(input_LDR_high_, gamma)/expo[2]

          gt_HDR_tm = np.log(1.0+5000.0*gt_HDR_) / np.log(1.0+5000.0)
          cv2.imwrite('%s/%03d_ours2_gt.hdr' % (result_path, n), gt_HDR_[:,:,::-1])
          cv2.imwrite('%s/%03d_ours2_gt_tm.png' % (result_path, n), 255.*gt_HDR_tm[:,:,::-1])

      
          batch_input_LDR_low = []
          batch_input_LDR_mid = []
          batch_input_LDR_high = []
          batch_input_HDR_low = []
          batch_input_HDR_mid = []
          batch_input_HDR_high = []
      
          input_LDR_low_ = input_LDR_low_*2. -1.
          input_LDR_mid_ = input_LDR_mid_*2. -1.
          input_LDR_high_ = input_LDR_high_*2. -1.
          input_HDR_low_ = input_HDR_low_ *2. -1.
          input_HDR_mid_ = input_HDR_mid_ *2. -1.
          input_HDR_high_ = input_HDR_high_ *2. -1.

          batch_input_LDR_low.append(input_LDR_low_)
          batch_input_LDR_mid.append(input_LDR_mid_)
          batch_input_LDR_high.append(input_LDR_high_)
          batch_input_HDR_low.append(input_HDR_low_)
          batch_input_HDR_mid.append(input_HDR_mid_)
          batch_input_HDR_high.append(input_HDR_high_)

          batch_input_LDR_low_tensor = np.stack(batch_input_LDR_low, axis=0)
          batch_input_LDR_mid_tensor = np.stack(batch_input_LDR_mid, axis=0)
          batch_input_LDR_high_tensor = np.stack(batch_input_LDR_high, axis=0)
          batch_input_HDR_low_tensor = np.stack(batch_input_HDR_low, axis=0)
          batch_input_HDR_mid_tensor = np.stack(batch_input_HDR_mid, axis=0)
          batch_input_HDR_high_tensor = np.stack(batch_input_HDR_high, axis=0)
          
          st = time.time()         
          recon_hdr = sess.run(output_hdr, feed_dict={
                                input_LDR_low    : batch_input_LDR_low_tensor,
                                input_LDR_mid    : batch_input_LDR_mid_tensor,
                                input_LDR_high   : batch_input_LDR_high_tensor,
                                input_HDR_low    : batch_input_HDR_low_tensor,
                                input_HDR_mid    : batch_input_HDR_mid_tensor,
                                input_HDR_high   : batch_input_HDR_high_tensor,
                                is_train         : is_training})
          ed = time.time() 
          
          img_recon_hdr_ = recon_hdr[0,:,:,:].astype(np.float32)
          img_recon_hdr_ += 1.
          img_recon_hdr_ /= 2.

          img_recon_hdr_tm = np.log(1.0+5000.0*img_recon_hdr_) / np.log(1.0+5000.0)
          cv2.imwrite('%s/%03d_ours2_hdr.hdr' % (result_path, n), img_recon_hdr_[:,:,::-1])
          cv2.imwrite('%s/%03d_ours2_hdr_tm.png' % (result_path, n), 255.*img_recon_hdr_tm[:,:,::-1])

          mseT = np.mean((gt_HDR_tm-img_recon_hdr_tm)**2)
          mseL = np.mean((gt_HDR_-img_recon_hdr_)**2)
          psnrT = -10.*math.log10(mseT)
          psnrL = -10.*math.log10(mseL)
          ssimT = ssim(gt_HDR_tm,img_recon_hdr_tm,multichannel=True,data_range=gt_HDR_tm.max() - gt_HDR_tm.min())
          ssimL = ssim(gt_HDR_,img_recon_hdr_,multichannel=True,data_range=gt_HDR_.max() - gt_HDR_.min())
          total_psnrT += psnrT
          total_psnrL += psnrL
          total_ssimT += ssimT
          total_ssimL += ssimL
          print('Iteration %06d Scene %03d psnrT is %f (%.4f)' % (iteration,n,psnrT,ed-st))

        print('Iteration %06d Total psnrT is %f.' % (iteration, total_psnrT/nscene))
        print('Iteration %06d Total psnrL is %f.' % (iteration, total_psnrL/nscene))
        print('Iteration %06d Total ssimT is %f.' % (iteration, total_ssimT/nscene))
        print('Iteration %06d Total ssimL is %f.' % (iteration, total_ssimL/nscene))
            
