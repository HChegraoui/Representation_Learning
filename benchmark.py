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
    tab =  np.random.randint(len(X2), size=10)
    for i in range(len(tab)):
        a,b,c = X2[tab[i]].shape
        X_Train.append(np.reshape(X2[tab[i]],a*b*c))
        Y_Train.append(Y2[tab[i]])
X= np.asarray(X_Train)
Y = np.asarray(Y_Train)
clf = svm.SVC(kernel='rbf', C = 1)
clf.fit(X,Y)

file = open("annotation_test.txt","r")
X =[]
Y =[]
i = 0
x = 0
for line in file:
    if(i%100== 0  and i> 0):
        X= np.asarray(X)
        Y = np.asarray(Y)
        Y_pred = clf.predict(X)
        for j in range(len(Y)):
            if(Y[j]==Y_pred[j]):
                x = x+1
        print(x)
        X =[]
        Y =[]
    sl = line.split(' ')
    Y.append(int(sl[1][0]))
    img  = cv2.imread("output/test/"+sl[0])/255
    a,b,c = img.shape
    X.append(np.reshape(img,a*b*c))
    i=i+1

    
