import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
import math
from sklearn import svm
from random import randint

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
def data_res(data):
    num = len(data)
    data_img = []
    idx = np.arange(0 , len(data))
    idx = idx[:num]
    for i in idx:
        img  = cv2.imread(data[ i], 3)/250        
        data_img.append(img)       
    data_img = np.asarray(data_img)
    return data_img
filenames =[]
for file in os.listdir("output/train/"):
    filenames.append("output/train/"+file)
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

saver.restore(sess, "./model_weights/encode_model_msda")
print("Model restored.")
Y = []
X=  []
file = open("annotation_train.txt","r")
for line in file:
    sl = line.split(' ')
    Y.append(int(sl[1][0]))
    img  = cv2.imread("output/train/"+sl[0])/255
    X.append(img)
X= np.asarray(X)
Y = np.asarray(Y)
X_Train =[]
Y_Train =[]
for i in range(3):
    X2 = X[np.where(Y==i)]
    Y2= Y[np.where(Y==i)]
    tab =  np.random.randint(len(X2), size=100)
    for i in range(len(tab)):
        X_Train.append(X2[tab[i]])
        Y_Train.append(Y2[tab[i]])
X= np.asarray(X_Train)
Y = np.asarray(Y_Train)
imgs = X.reshape((-1, 32, 32, 3))
x_test_noisy = add_noise(imgs)
recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
units = sess.run(encoded,feed_dict={inputs_:x_test_noisy})
####"
print("svm start")
print("SVM TRAIN")
clf = svm.SVC(kernel='rbf', C = 10)
clf.fit(units.reshape(len(units),-1),Y)
file = open("annotation_test.txt","r")
X =[]
Y =[]
i = 0
x = 0
for line in file:
    if(i%100== 0  and i> 0):
        X= np.asarray(X)
        Y = np.asarray(Y)
        imgs = X.reshape((-1, 32, 32, 3))
        x_test_noisy = add_noise(imgs)
        recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
        units_test = sess.run(encoded,feed_dict={inputs_:x_test_noisy})
        Y_pred = clf.predict(units_test.reshape(len(units_test),-1))
        for j in range(len(Y)):
            if(Y[j]==Y_pred[j]):
                x = x+1
        print(x)
        X =[]
        Y =[]
    sl = line.split(' ')
    Y.append(int(sl[1][0]))
    img  = cv2.imread("output/test/"+sl[0])/255
    X.append(img)
    i=i+1
print("test")
print(x)
sess.close()
sys.exit()
for i in range(10):
    im1 = cv2.resize(imgs[i], (300, 300))
    im2 = cv2.resize(recon_img[i], (300, 300))
    im3 = cv2.resize(x_test_noisy[i], (300, 300)) 
    cv2.imshow('Original', im1)
    cv2.imshow('Reconst', im2)
    cv2.imshow('Noisi', im3)
    for j in range(32):
        imk = cv2.resize(units[i,: ,: ,j], (300, 300))
        cv2.imshow("hidden"+str(j), imk)
    cv2.waitKey()
    sys.exit()

sess.close()
