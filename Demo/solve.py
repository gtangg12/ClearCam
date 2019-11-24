import cv2 as cv
import numpy as np
from keras import backend as K
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.optimizers import *
from keras.metrics import *
from keras.utils import *
import glob
import random

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def SSIM(y_true, y_pred):
    ux = np.mean(y_pred)
    uy = np.mean(y_true)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (2*ux*uy+c1)*(2*np.std(y_pred)*np.std(y_true)+c2)
    denom =(ux**2+uy**2+c1)*(np.var(y_pred)+np.var(y_true)+c2)
    return ssim/np.clip(denom, 0.0000001, np.inf)

#image_names = [file for file in glob.glob('HazyImages/*.png')]
image_names = ['rocks']#, 'lampshade']

MODEL_NAME = "regressor3"
model_json_name = "../final_model/" + MODEL_NAME + ".json"
model_weights_name = "../final_model/" + MODEL_NAME + ".h5"

json_file = open(model_json_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_weights_name)
loaded_model.compile(optimizer='adam', loss='mae')
print("Loaded model from disk")

cnt, hmae, hssim, mae, ssim = 0, 0, 0, 0, 0

for name in image_names[:1]:
    dis = cv.imread('dismaps/'+name+'?0.png', 0)/3.0
    cnt+=1
    #fill in gaps
    for i in range(1, dis.shape[0]):
        for j in range(dis.shape[1]):
            if (dis[i,j] == 0):
                dis[i, j] = dis[i-1, j]
    for i in range(dis.shape[0]):
        for j in range(1, dis.shape[1]):
            if (dis[i,j] == 0):
                dis[i, j] = dis[i, j-1]
    dis = np.clip(dis, 1, dis.shape[1])
    #compute depth from dismap
    depth = 3740*0.016/dis
    #solve for dehazed image
    clearimg = cv.imread(name+"_c.png", 1)/255.
    I = cv.imread(name+"_0_1.png", 1)/255.
    A = np.max(I)
    B = loaded_model.predict(np.array([I[128:256,128:256]]))
    print("Predicted Alpha: "+str(A))
    print("Predicted Beta: "+str(B[0,0]))
    T = np.exp(-B*cv.merge((depth, depth, depth)))
    J = np.divide((I-A*(1-T)),T)
    I = np.clip(I, 0, 1)
    J = np.clip(J, 0, 1)
    hssim+=SSIM(clearimg, I)
    ssim+=SSIM(clearimg, J)
    I*=255
    J*=255
    I = I.astype(np.uint8)
    J = J.astype(np.uint8)
    hmae+=MAE(clearimg, I)
    mae+=MAE(clearimg, J)
    cv.imshow('Hazy_Image', I)
    cv.imshow('Clear_Image', clearimg)
    cv.imshow('Our_Result', J)
    cv.waitKey(0)

#print(hmae/cnt)
print("HazySSIM: "+str(hssim/cnt))
#print(mae/cnt)
print("DehzedSSIM: "+str(ssim/cnt))
#print(cnt)
