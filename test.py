import cv2
import numpy as np        
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

model = load_model('model2.h5')

def predict_digit(img):
    test_image = img.reshape(-1,28,28,1)
    test_image = test_image / 255
    pred = model.predict([test_image])
    print(pred)
    return np.argmax(pred)

def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray

img = cv2.imread('a.png',0)
print(np.shape(img))
plt.imshow(img)
plt.show()
'''
img = cv2.bitwise_not(img)
img = image_refiner(img)
plt.imshow(img)
plt.show()

th,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(img)
plt.show()


dbfile = open('data.pkl', 'rb')      
img = pickle.load(dbfile) 
dbfile.close() 

plt.imshow(img)
plt.show()
'''
value = predict_digit(img)
print(value)