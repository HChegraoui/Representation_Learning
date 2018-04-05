import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
import math

random.seed(10)
def add_noise(data):
    data_2 = np.copy(data)
    pr = 0.3
    for img  in data_2:
        x ,  y , z = img.shape
        max_x = x//8 - 1 
        max_y = y//8 -1
        for i in range(max_x ):
            for j in range( max_y):
                r = random.uniform(0, 1)
                if(r < pr ):
                    img[8*i:8*(i+1) , 8*j:8*(j+1),:]  = 0
    return data_2
def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data_img = []
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    for i in idx:
        img  = cv2.imread(data[ i], 3)/250        
        data_img.append(img)
        
    data_img = np.asarray(data_img)
    return data_img

filenames =[]
for file in os.listdir("output/test/"):
    filenames.append("output/test/"+file)
print(len(filenames))
inputs_ = tf.placeholder(tf.float32,[None,32,32,3])
targets_ = tf.placeholder(tf.float32,[None,32,32,3])
def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)
with tf.name_scope('en-convolutions'):
    conv1 = tf.layers.conv2d(inputs_,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv1')
# Now 32x32x32
with tf.name_scope('en-pooling'):
    maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
# Now 16x16x32
with tf.name_scope('en-convolutions'):
    conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2')
# Now 14x14x32
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='encoding')
# Now 7x7x32.
#latent space

with tf.name_scope('decoder'):
    conv3 = tf.layers.conv2d(encoded,filters=32,kernel_size=(3,3),strides=(1,1),name='conv3',padding='SAME',use_bias=True,activation=lrelu)
#Now 7x7x32        
    upsample1 = tf.layers.conv2d_transpose(conv3,filters=32,kernel_size=3,padding='same',strides=2,name='upsample1')
# Now 16x16x32
    upsample2 = tf.layers.conv2d_transpose(upsample1,filters=32,kernel_size=3,padding='same',strides=2,name='upsample2')
# Now 28x28x32
    logits = tf.layers.conv2d(upsample2,filters=3,kernel_size=(3,3),strides=(1,1),name='logits',padding='SAME',use_bias=True)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
    decoded = tf.sigmoid(logits,name='recon')
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=targets_)
saver = tf.train.Saver()

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
sess = tf.Session()

saver.restore(sess, "./model_weights/encode_model_sparsec1c2")
print("Model restored.")
batch_x = next_batch(10 , filenames)
imgs = batch_x.reshape((-1, 32, 32, 3))
#noise_factor = 0.5
#x_test_noisy = add_noise(imgs)
recon_img = sess.run([decoded], feed_dict={inputs_: imgs})[0]
units = sess.run(conv2,feed_dict={inputs_:imgs})
units2 = sess.run(conv1,feed_dict={inputs_:imgs})
enc =  sess.run(encoded,feed_dict={inputs_:imgs})
#sys.exit()
for i in range(10):
    im1 = cv2.resize(imgs[i], (300, 300))
    im2 = cv2.resize(recon_img[i], (300, 300))
    #im3 = cv2.resize(x_test_noisy[i], (300, 300))
    cv2.imwrite( 'Original_'+str(i)+'.jpg' ,im1*254)
    cv2.imwrite( 'Recon_'+str(i)+'.jpg' ,im2*254)
    j = 0
    for r in range(4):
        imr = cv2.resize(units[i,: ,: ,j], (100, 100))
        j = j+1
        for c in range(1,8):
            imj = cv2.resize(units[i,: ,: ,j], (100, 100))
            imr = np.hstack((imr, imj))
            j = j+1
        if r == 0:
            im3 = imr
        else :
            im3 = np.vstack((im3, imr))
    cv2.imwrite( 'Mixed_'+str(i)+'.jpg' ,im3*254)
    j= 0
    for r in range(4):
        imr = cv2.resize(units2[i,: ,: ,j], (100, 100))
        j = j+1
        for c in range(1,8):
            imj = cv2.resize(units2[i,: ,: ,j], (100, 100))
            imr = np.hstack((imr, imj))
            j = j+1
        if r == 0:
            im3 = imr
        else :
            im3 = np.vstack((im3, imr))
    cv2.imwrite( 'Mixed_2'+str(i)+'.jpg' ,im3*254)
    j = 0
    for r in range(4):
        imr = cv2.resize(enc[i,: ,: ,j], (100, 100))
        j = j+1
        for c in range(1,8):
            imj = cv2.resize(enc[i,: ,: ,j], (100, 100))
            imr = np.hstack((imr, imj))
            j = j+1
        if r == 0:
            im3 = imr
        else :
            im3 = np.vstack((im3, imr))
    cv2.imwrite( 'encoded'+str(i)+'.jpg' ,im3*254)
    cv2.waitKey()
    sys.exit()

sess.close()
