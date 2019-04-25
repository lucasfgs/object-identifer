# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:52:52 2019

@author: Lucas Ferreira
"""
import sys
from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt

#%config.Inlinebackend.figure_format = 'svg' 
#Could not create cuDNN handle when convnets are 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


options = {
        
        'model':'cfg/yolo.cfg',
        'load':'bin/yolov2.weights',
        'threshold':0.3,
        'gpu': 1.0
        }

tfnet = TFNet(options)  #print modlw arch

img = cv2.imread('pecuaria.jpg')
img.shape

result = tfnet.return_predict(img)
resultLength = len(result)

for i in range(resultLength):    
    t1 = (result[i]['topleft']['x'],result[i]['topleft']['y']) #top left
    b1 = (result[i]['bottomright']['x'],result[i]['bottomright']['y']) # bottomright

    labl = result[i]['label']
    img = cv2.rectangle(img, t1, b1,(0.,255,0),6)    
    #cv2.putText(img,labl, t1,cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.putText(img, "Counter: " + str(resultLength), (10,150),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10)

plt.imshow(img)
plt.savefig('img-output.jpg', dpi=300, quality=95)