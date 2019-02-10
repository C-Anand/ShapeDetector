
# coding: utf-8

# In[2]:

from DataGeneration import *
from TestTrainAndModel import model


# In[9]:




# In[13]:


from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from scipy import misc
import cv2
#matplotlib inline

 
path  = input('Enter the path of image\n')



img = cv2.imread(path)
img = cv2.resize(img,(128,128),interpolation = cv2.INTER_AREA)

imgfeatures = img.reshape(1, img.shape[0], img.shape[1],img.shape[2])

class_probabilities = model.predict(imgfeatures)

class_idx = np.argmax(class_probabilities, axis=1)
print(class_idx)
print(classnames[int(class_idx[0])])


