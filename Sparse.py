import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random
def kl_divergence(p, p_hat):
        print(p_hat.shape)
        return tf.reduce_mean(p * tf.log(p) - p * tf.log(tf.reduce_mean(p_hat)) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - tf.reduce_mean(p_hat)))
def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data_img = []
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    for i in idx:
        img  = cv2.imread(data[ i])/255        
        data_img.append(img)
    data_img = np.asarray(data_img)
    return data_img

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

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
'''
reading data set of images
'''

filenames =[]
for file in os.listdir("output/all/"):
    filenames.append("output/all/"+file)
''' Input data and target data '''
inputs_ = tf.placeholder(tf.float32,[None,32,32,3])
targets_ = tf.placeholder(tf.float32,[None,32,32,3])
''' Building model '''
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
loss =  tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(targets_, logits), 2.0),0))+0.5* kl_divergence(0.1, (tf.reduce_mean(conv1,0)))+0.5* kl_divergence(0.1, (tf.reduce_mean(conv2,0)))

learning_rate=tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  #cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer

sess = tf.Session()    

''' train parameters '''
saver = tf.train.Saver()
loss = []
valid_loss = []
noise_factor = 0.5


display_step = 1
epochs = 25
batch_size = 64
lr=1e-5
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./graphs', sess.graph)
total_batch = int(len(filenames)/batch_size)
print("total batch : " + str(total_batch))
''' Start Training '''

for e in range(epochs):
    print("epoche number : " +str(e+1))
    for ibatch in range(total_batch):
        batch_x = next_batch(batch_size , filenames)
        imgs = batch_x.reshape((-1, 32, 32, 3))
        imgs_test = batch_x.reshape((-1, 32, 32,3 ))
        #x_train_noisy = add_noise(imgs)
        #x_test_noisy = add_noise(imgs_test)
        
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                         targets_: imgs,learning_rate:lr})
      
        batch_cost_test = sess.run(cost, feed_dict={inputs_: imgs_test,
                                                         targets_: imgs_test})
    if (e+1) % display_step == 0:
        print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost),
                 "Validation loss: {:.4f}".format(batch_cost_test))
   
    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
print("end of training")
saver.save(sess, './model_weights/encode_model_sparsec1c2') 
writer.close()
sess.close()
