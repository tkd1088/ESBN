
import numpy as np
import os
import tensorflow as tf
import cv2
from random import shuffle
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_DEPTH = 3
STRIDE = 64
gamma = 2.2

is_shuffle = True
files_per_tfrecord = 1000

dataset_path = "/data/tkd1088/dataset/Kalantari/Training"

nscene = 74
nlow = 12
ncol = 20
nrot = 4
nflip = 2
npatch = nscene*nlow*ncol*nrot*nflip

n = np.arange(npatch)
np.random.shuffle(n)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

i = 0
j = 0
while i < npatch :
	n_tfrecord = i // files_per_tfrecord
	train_filename = "/data/tkd1088/dataset/Kalantari/tfrecord_190412/train_%03d.tfrecords" % n_tfrecord
	 
	if not i % files_per_tfrecord:
		print ('Train data: %d/%d. %s is generated!' % (i, len(n), train_filename))

	with tf.python_io.TFRecordWriter(train_filename) as writer :
		j = 0
		while i < npatch and j < files_per_tfrecord :
			s = n[i]//(nlow*ncol*nrot*nflip)
			h = (n[i]-nlow*ncol*nrot*nflip*s)//(ncol*nrot*nflip)
			w = (n[i]-nlow*ncol*nrot*nflip*s-ncol*nrot*nflip*h)//(nrot*nflip)
			r = (n[i]-nlow*ncol*nrot*nflip*s-ncol*nrot*nflip*h-nrot*nflip*w)//nflip
			f = n[i]-nlow*ncol*nrot*nflip*s-ncol*nrot*nflip*h-nrot*nflip*w-nflip*r

			scene_path = '%s/%03d' % (dataset_path,s+1)
			image_pathss = '%s/*.tif' % scene_path
			image_paths = sorted(glob.glob(image_pathss))
			info_path = '%s/exposure.txt' % scene_path
			hdr_path = '%s/HDRImg.hdr' % scene_path

			input_LDR_low = cv2.imread(image_paths[0]).astype(np.float32)/255.
			input_LDR_mid = cv2.imread(image_paths[1]).astype(np.float32)/255.
			input_LDR_high = cv2.imread(image_paths[2]).astype(np.float32)/255.
			gt_HDR = cv2.imread(hdr_path,-1).astype(np.float32)

			height, width, channel = input_LDR_low.shape

			expo = np.zeros(3)
			file = open(info_path, 'r')
			expo[0]= np.power(2.0, float(file.readline()))
			expo[1]= np.power(2.0, float(file.readline()))
			expo[2]= np.power(2.0, float(file.readline()))
			file.close()
			#in_exps = np.array(open(info_path).read().split('\n')[:3]).astype(np.float32)

			input_LDR_low_patch = input_LDR_low[h*STRIDE:h*STRIDE+IMG_HEIGHT, w*STRIDE:w*STRIDE+IMG_WIDTH,::-1]
			input_LDR_mid_patch = input_LDR_mid[h*STRIDE:h*STRIDE+IMG_HEIGHT, w*STRIDE:w*STRIDE+IMG_WIDTH,::-1]
			input_LDR_high_patch = input_LDR_high[h*STRIDE:h*STRIDE+IMG_HEIGHT, w*STRIDE:w*STRIDE+IMG_WIDTH,::-1]
			gt_HDR_patch = gt_HDR[h*STRIDE:h*STRIDE+IMG_HEIGHT, w*STRIDE:w*STRIDE+IMG_WIDTH,::-1]

			input_HDR_low_patch = np.power(input_LDR_low_patch, gamma)/expo[0]
			input_HDR_mid_patch = np.power(input_LDR_mid_patch, gamma)/expo[1]
			input_HDR_high_patch = np.power(input_LDR_high_patch, gamma)/expo[2]

			gt_LDR_low_patch = np.clip(gt_HDR_patch*expo[0], 0.0, 1.0)
			gt_LDR_mid_patch = np.clip(gt_HDR_patch*expo[1], 0.0, 1.0)
			gt_LDR_high_patch = np.clip(gt_HDR_patch*expo[2], 0.0, 1.0)
			gt_LDR_low_patch = np.power(gt_LDR_low_patch, 1./gamma)
			gt_LDR_mid_patch = np.power(gt_LDR_mid_patch, 1./gamma)
			gt_LDR_high_patch = np.power(gt_LDR_high_patch, 1./gamma)
	
			gt_HDR_low_patch = np.power(gt_LDR_low_patch, gamma)/expo[0]
			gt_HDR_mid_patch = np.power(gt_LDR_mid_patch, gamma)/expo[1]
			gt_HDR_high_patch = np.power(gt_LDR_high_patch, gamma)/expo[2]

			input_LDR_low_patch = (255.*input_LDR_low_patch).astype(np.uint8)
			input_LDR_mid_patch = (255.*input_LDR_mid_patch).astype(np.uint8)
			input_LDR_high_patch = (255.*input_LDR_high_patch).astype(np.uint8)
			gt_LDR_low_patch = (255.*gt_LDR_low_patch).astype(np.uint8)
			gt_LDR_mid_patch = (255.*gt_LDR_mid_patch).astype(np.uint8)
			gt_LDR_high_patch= (255.*gt_LDR_high_patch).astype(np.uint8)


			input_LDR_low_patch = cv2.resize(input_LDR_low_patch, (250,250))
			input_LDR_mid_patch = cv2.resize(input_LDR_mid_patch, (250,250))
			input_LDR_high_patch = cv2.resize(input_LDR_high_patch, (250,250))
			input_HDR_low_patch = cv2.resize(input_HDR_low_patch, (250,250))
			input_HDR_mid_patch = cv2.resize(input_HDR_mid_patch, (250,250))
			input_HDR_high_patch = cv2.resize(input_HDR_high_patch, (250,250))
			gt_LDR_low_patch = cv2.resize(gt_LDR_low_patch, (250,250))
			gt_LDR_mid_patch = cv2.resize(gt_LDR_mid_patch, (250,250))
			gt_LDR_high_patch = cv2.resize(gt_LDR_high_patch, (250,250))
			gt_HDR_low_patch = cv2.resize(gt_HDR_low_patch, (250,250))
			gt_HDR_mid_patch = cv2.resize(gt_HDR_mid_patch, (250,250))
			gt_HDR_high_patch = cv2.resize(gt_HDR_high_patch, (250,250))
			gt_HDR_patch = cv2.resize(gt_HDR_patch, (250,250))

			input_LDR_low_patch = cv2.resize(input_LDR_low_patch, (256,256))
			input_LDR_mid_patch = cv2.resize(input_LDR_mid_patch, (256,256))
			input_LDR_high_patch = cv2.resize(input_LDR_high_patch, (256,256))
			input_HDR_low_patch = cv2.resize(input_HDR_low_patch, (256,256))
			input_HDR_mid_patch = cv2.resize(input_HDR_mid_patch, (256,256))
			input_HDR_high_patch = cv2.resize(input_HDR_high_patch, (256,256))
			gt_LDR_low_patch = cv2.resize(gt_LDR_low_patch, (256,256))
			gt_LDR_mid_patch = cv2.resize(gt_LDR_mid_patch, (256,256))
			gt_LDR_high_patch = cv2.resize(gt_LDR_high_patch, (256,256))
			gt_HDR_low_patch = cv2.resize(gt_HDR_low_patch, (256,256))
			gt_HDR_mid_patch = cv2.resize(gt_HDR_mid_patch, (256,256))
			gt_HDR_high_patch = cv2.resize(gt_HDR_high_patch, (256,256))
			gt_HDR_patch = cv2.resize(gt_HDR_patch, (256,256))


			M = cv2.getRotationMatrix2D((IMG_WIDTH//2,IMG_HEIGHT//2), 90*r, 1.0)
			input_LDR_low_patch = cv2.warpAffine(input_LDR_low_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			input_LDR_mid_patch = cv2.warpAffine(input_LDR_mid_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			input_LDR_high_patch = cv2.warpAffine(input_LDR_high_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			input_HDR_low_patch = cv2.warpAffine(input_HDR_low_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			input_HDR_mid_patch = cv2.warpAffine(input_HDR_mid_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			input_HDR_high_patch = cv2.warpAffine(input_HDR_high_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_LDR_low_patch = cv2.warpAffine(gt_LDR_low_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_LDR_mid_patch = cv2.warpAffine(gt_LDR_mid_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_LDR_high_patch = cv2.warpAffine(gt_LDR_high_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_HDR_low_patch = cv2.warpAffine(gt_HDR_low_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_HDR_mid_patch = cv2.warpAffine(gt_HDR_mid_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_HDR_high_patch = cv2.warpAffine(gt_HDR_high_patch, M, (IMG_WIDTH, IMG_HEIGHT))
			gt_HDR_patch = cv2.warpAffine(gt_HDR_patch, M, (IMG_WIDTH, IMG_HEIGHT))

			if f > 0 :
				input_LDR_low_patch = input_LDR_low_patch[:,::-1,:]
				input_LDR_mid_patch = input_LDR_mid_patch[:,::-1,:]
				input_LDR_high_patch = input_LDR_high_patch[:,::-1,:]
				input_HDR_low_patch = input_HDR_low_patch[:,::-1,:]
				input_HDR_mid_patch = input_HDR_mid_patch[:,::-1,:]
				input_HDR_high_patch = input_HDR_high_patch[:,::-1,:]
				gt_LDR_low_patch = gt_LDR_low_patch[:,::-1,:]
				gt_LDR_mid_patch = gt_LDR_mid_patch[:,::-1,:]
				gt_LDR_high_patch = gt_LDR_high_patch[:,::-1,:]
				gt_HDR_low_patch = gt_HDR_low_patch[:,::-1,:]
				gt_HDR_mid_patch = gt_HDR_mid_patch[:,::-1,:]
				gt_HDR_high_patch = gt_HDR_high_patch[:,::-1,:]
				gt_HDR_patch = gt_HDR_patch[:,::-1,:]


			feature = {'train/input_LDR_low'	: _bytes_feature(input_LDR_low_patch.tostring()),
						'train/input_LDR_mid'	: _bytes_feature(input_LDR_mid_patch.tostring()),
						'train/input_LDR_high'	: _bytes_feature(input_LDR_high_patch.tostring()),
						'train/input_HDR_low'	: _bytes_feature(input_HDR_low_patch.tostring()),
						'train/input_HDR_mid'	: _bytes_feature(input_HDR_mid_patch.tostring()),
						'train/input_HDR_high'	: _bytes_feature(input_HDR_high_patch.tostring()),
    					'train/gt_LDR_low'		: _bytes_feature(gt_LDR_low_patch.tostring()),
    					'train/gt_LDR_high'		: _bytes_feature(gt_LDR_high_patch.tostring()),
    					'train/gt_HDR'			: _bytes_feature(gt_HDR_patch.tostring())}

			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

			print("%d / %d patches are converted!" % (i, npatch))

					#cv2.imwrite('/data/tkd1088/dataset/Kalantari/tfrecord_190402/tmp_low.png', np.power((expo[2]*gt_HDR_high),1./gamma)*255.)
					#cv2.imwrite('/data/tkd1088/dataset/Kalantari/tfrecord_190402/tmp_low2.png', gt_HDR*255.)
					#cv2.imwrite('/data/tkd1088/dataset/Kalantari/tfrecord_190402/tmp_low3.png', gt_HDR_mid*255.)

			i += 1
			j += 1
