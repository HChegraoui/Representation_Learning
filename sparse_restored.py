import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)
def next_batch(num, data , labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data_img = []
    label = []
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    for i in idx:
        img  = cv2.imread(data[ i])/255        
        data_img.append(img)
        label.append(labels[i])
    data_img = np.asarray(data_img)
    label =  np.asarray(label)
    return (data_img , label )
Y = []
X=  []
file  = open('annotation_test.txt')
for line in file:
    sl = line.split(' ')
    v = int(sl[1])
    X.append("output/test/" + sl[0])
    Y.append(v)
classes = 3 
inputs_ = tf.placeholder(tf.float32,[None,32,32,3])
targets_ =tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(targets_, classes)

''' Building model '''
with tf.name_scope('en-convolutions'):
    conv1 = tf.layers.conv2d(inputs_,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv1')
# Now 32x32x32
with tf.name_scope('en-pooling'):
    maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
# Now 16x16x32
with tf.name_scope('en-convolutions'):
    conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2')
# Now 16x16x32
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='encoding')
# Now 8x8x32.
#latent space
with tf.name_scope("flattening"):
    flat=  tf.contrib.layers.flatten(encoded)
with tf.name_scope("fully_connected"):
    connected = tf.contrib.layers.fully_connected(flat , 500)
with tf.name_scope("fully_connected_reg"):
    final =  tf.contrib.layers.fully_connected(flat , classes)
learning_rate=tf.placeholder(tf.float32)
loss = tf.nn.softmax_cross_entropy_with_logits(logits= final, labels = one_hot_y)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
sess = tf.Session() 

saver = tf.train.Saver()
saver.restore(sess, "./model_weights/end")
print("Model restored.")
noise_factor = 0.5
total = 0
count  = 0
for j in range(100):
    X_test , Y_test = next_batch(100 , X , Y )

    units = sess.run(final,feed_dict={inputs_:X_test})
    num = 0
    for i in range(len(units)):
        count = count +1 
        if(np.argmax(units[i]) == Y[i]):
            num = num +1
            total = total +1 
    print(num)
print( total)
print(count)

