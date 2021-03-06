
import network_align2
import numpy as np
import tensorflow as tf
import os
import cv2
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_DEPTH = 3

data_path = '/data/tkd1088/dataset/Kalantari/tfrecord_190506/*.tfrecords'
data_path_addrs = glob.glob(data_path)

result_path = '/data/tkd1088/result/hdr_transfer/result_190510_1_high'
if not os.path.exists(result_path):
  os.mkdir(result_path)
ckpt_path = '/data/tkd1088/result/hdr_transfer/ckpt_190510_1_high'
if not os.path.exists(ckpt_path):
  os.mkdir(ckpt_path)

batch_size = 8;
init_learning_rate = 1e-4
step_lr_adjust = 200000
epsilon = 1e-8 # AdamOptimizer epsilon
nesterov_momentum = 0.9
weight_decay = 0
start_i = 0
iteration = 300000
step_disp_train = 1000
step_test_save = 1000
step_ckpt_save = 10000

is_training = True

expo_times = tf.placeholder(tf.float32, shape=[None, 3])
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
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

my_CAE = network_align2.MCAE_align(input_LDR_low, input_LDR_mid, input_HDR_low, input_HDR_mid, is_train)
output_low = my_CAE.output_low
#output_high = my_CAE.output_high
#output_hdr = my_CAE.output_hdr

loss_low = tf.reduce_mean(tf.square(tf.subtract(output_low,gt_LDR_low)))
#loss_high = tf.reduce_mean(tf.square(tf.subtract(output_high,gt_LDR_high)))
#output_hdr_tm = tf.log(1 + 5000.* (output_hdr+1)/2.) / tf.log(1 +5000.) *2. -1
#gt_hdr_tm = tf.log(1 + 5000.* (gt_hdr+1)/2.) / tf.log(1 +5000.) *2. -1
#loss_hdr = tf.reduce_mean(tf.square(tf.subtract(output_hdr_tm,gt_hdr_tm)))
#loss = loss_low + loss_high + 1*loss_hdr
loss = loss_low

regularization = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=epsilon)
  train = optimizer.minimize(loss + regularization * weight_decay)


# In[7]: make batch

filename_queue = tf.train.string_input_producer(data_path_addrs)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

feature = {'train/input_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR': tf.FixedLenFeature([], tf.string)}

features = tf.parse_single_example(serialized_example, features=feature)

inLl = tf.decode_raw(features['train/input_LDR_low'], tf.uint8)
inLm = tf.decode_raw(features['train/input_LDR_mid'], tf.uint8)
inLh = tf.decode_raw(features['train/input_LDR_high'], tf.uint8)
inHl = tf.decode_raw(features['train/input_HDR_low'], tf.float32)
inHm = tf.decode_raw(features['train/input_HDR_mid'], tf.float32)
inHh = tf.decode_raw(features['train/input_HDR_high'], tf.float32)
gtLl = tf.decode_raw(features['train/gt_LDR_low'], tf.uint8)
gtLh = tf.decode_raw(features['train/gt_LDR_high'], tf.uint8)
gtHm = tf.decode_raw(features['train/gt_HDR'], tf.float32)

inLl = tf.reshape(inLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
inLm = tf.reshape(inLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
inLh = tf.reshape(inLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
inHl = tf.reshape(inHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
inHm = tf.reshape(inHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
inHh = tf.reshape(inHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gtLl = tf.reshape(gtLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gtLh = tf.reshape(gtLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
gtHm = tf.reshape(gtHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

# normalize to [-1,+1]
inLl = tf.cast(inLl, tf.float32)/127.5 - 1.
inLm = tf.cast(inLm, tf.float32)/127.5 - 1.
inLh = tf.cast(inLh, tf.float32)/127.5 - 1.
inHl = tf.cast(inHl, tf.float32)*2. - 1.
inHm = tf.cast(inHm, tf.float32)*2. - 1.
inHh = tf.cast(inHh, tf.float32)*2. - 1.
gtLl = tf.cast(gtLl, tf.float32)/127.5 - 1.
gtLh = tf.cast(gtLh, tf.float32)/127.5 - 1.
gtHm = tf.cast(gtHm, tf.float32)*2. - 1.

inLls, inLms, inLhs, inHls, inHms, inHhs, gtLls, gtLhs, gtHms = tf.train.shuffle_batch([inLl, inLm, inLh, inHl, inHm, inHh, gtLl, gtLh, gtHm], batch_size=batch_size, capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)

# In[8]: training session
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

train_loss_total = 0
train_loss_low_total = 0
train_loss_high_total = 0
train_loss_hdr_total = 0


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if start_i > 0 :
      saver.restore(sess, '%s/iter_%06d' % (ckpt_path, start_i))
      print('A checkpoint file %s/iter_%06d.ckpt is loaded' % (ckpt_path, start_i))

    #summary_writer = tf.summary.FileWriter('/data/tkd1088/result/hdr-transfer/log', sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(start_i+1, iteration+1):

        bt_inLl, bt_inLm, bt_inLh, bt_inHl, bt_inHm, bt_inHh, bt_gtLl, bt_gtLh, bt_gtHm = sess.run([inLls, inLms, inLhs, inHls, inHms, inHhs, gtLls, gtLhs, gtHms])
        
        _, train_loss, train_loss_low, recon_low = sess.run([train, loss, loss_low, output_low], feed_dict={
                           input_LDR_low    : bt_inLh,
                           input_LDR_mid    : bt_inLm,
                           input_HDR_low    : bt_inHh,
                           input_HDR_mid    : bt_inHm,
                           gt_LDR_low       : bt_gtLh,
                           gt_HDR           : bt_gtHm,
                           is_train         : is_training,
                           learning_rate    : init_learning_rate})

        train_loss_total += train_loss
        train_loss_low_total += train_loss_low
        #train_loss_high_total += train_loss_high
        #train_loss_hdr_total += train_loss_hdr

        if i % step_disp_train == 0:
            print('[step % 6d] train loss : %.7f+%.7f+%.7f = %.7f (lr=%.1e)' % (i, train_loss_low_total/step_disp_train, train_loss_high_total/step_disp_train, train_loss_hdr_total/step_disp_train, train_loss_total/step_disp_train, init_learning_rate))
            train_loss_total = 0
            train_loss_low_total = 0
            train_loss_high_total = 0
            train_loss_hdr_total = 0
            
       
        if i % step_test_save == 0:
            img_recon_low = recon_low[0,:,:,::-1].astype(np.float32)
            #img_recon_high = recon_high[0,:,:,::-1].astype(np.float32)
            #img_recon_hdr = recon_hdr[0,:,:,::-1].astype(np.float32)
            #img_gt_low = bt_gtLl[0,:,:,::-1].astype(np.float32)
            img_gt_high = bt_gtLh[0,:,:,::-1].astype(np.float32)
            #img_gt_hdr = bt_gthm[0,:,:,::-1].astype(np.float32)
            #img_input_low = bt_inLl[0,:,:,::-1].astype(np.float32)
            img_input_mid = bt_inLm[0,:,:,::-1].astype(np.float32)
            img_input_high = bt_inLh[0,:,:,::-1].astype(np.float32)

            img_recon_low += 1.
            #img_recon_high += 1.
            #img_recon_hdr += 1.
            #img_gt_low += 1.
            img_gt_high += 1.
            #img_gt_hdr += 1.
            #img_input_low += 1.
            img_input_mid += 1.
            img_input_high += 1.

            img_recon_low *= 127.5
            #img_recon_high *= 127.5
            #img_recon_hdr /= 2.
            #img_gt_low *= 127.5
            img_gt_high *= 127.5
            #img_gt_hdr /= 2.
            #img_input_low *= 127.5
            img_input_mid *= 127.5
            img_input_high *= 127.5

            #img_gt_hdr_tm = 255. * np.log(1.0+5000.0*img_gt_hdr) / np.log(1.0+5000.0)
            #img_recon_hdr_tm = 255.* np.log(1.0+5000.0*img_recon_hdr) / np.log(1.0+5000.0)

            #result1 = np.concatenate((img_input_low, img_recon_low, img_gt_low), axis=0)
            result2 = np.concatenate((img_input_mid, img_input_mid, img_input_mid), axis=0)
            result3 = np.concatenate((img_input_high, img_recon_low, img_gt_high), axis=0)
            #result4 = np.concatenate((img_input_mid, img_recon_hdr_tm, img_gt_hdr_tm), axis=0)
            result = np.concatenate((result2, result3), axis=1)

            cv2.imwrite('%s/iter_%06d.png' % (result_path, i), result)
            print('A result file is saved in %s/iter_%06d.png' % (result_path, i))

        if i % step_ckpt_save == 0:
            save_path = saver.save(sess, '%s/iter_%06d' % (ckpt_path, i))
            print('A checkpoint file is saved in %s/iter_%06d.ckpt' % (ckpt_path, i))

        if i % step_lr_adjust == 0:
            init_learning_rate /= 10.0
            step_lr_adjust *= 2

    coord.request_stop()
    coord.join(threads)
    sess.close()


