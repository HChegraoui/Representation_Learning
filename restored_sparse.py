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
values =[0 , 0 , 0]
maxi = 1600
file  = open('annotation_train2.txt')
for line in file:
    sl = line.split(' ')
    v = int(sl[1])
    if(values[v] > maxi):
        continue
    else:
        X.append("output/train2/" + sl[0])
        Y.append(v)
        values[v] = values[v]+1
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
with tf.name_scope("fully_connected"):
    final =  tf.contrib.layers.fully_connected(flat , classes)
print(final.shape)
learning_rate=tf.placeholder(tf.float32)
loss = tf.nn.softmax_cross_entropy_with_logits(logits= final, labels = one_hot_y)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
sess = tf.Session() 
display_step = 1
epochs = 25
batch_size = 30
lr=1e-5
loss = []
valid_loss = []
sess.run(tf.global_variables_initializer())
total_batch = int(len(X)/batch_size)
variables = tf.contrib.slim.get_variables_to_restore()
variables_to_restore = [v for v in variables if v.name.split('/')[0].find("fully") < 0]
print("total batch : " + str(total_batch))
''' Start Training '''
saver = tf.train.Saver(variables_to_restore)
noise_factor = 0.5
saver.restore(sess, './model_weights/encode_model_sparsec1c2')

for e in range(epochs):
    print("epoche number : " +str(e+1))
    for ibatch in range(total_batch):
        batch_x , label_x = next_batch(batch_size , X,Y)
        imgs = batch_x.reshape((-1, 32, 32, 3))
        
        
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: label_x,learning_rate:lr})
      
        batch_cost_test = sess.run(cost, feed_dict={inputs_: imgs,
                                                         targets_: label_x})
    if (e+1) % display_step == 0:
        print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost),
                 "Validation loss: {:.4f}".format(batch_cost_test))
   
    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
saver = tf.train.Saver()
saver.save(sess, './model_weights/end') 

print("end of training")
sess.close()
