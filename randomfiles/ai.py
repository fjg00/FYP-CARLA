from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import time
import matplotlib.pyplot as plt
from threading import Thread
# define a function which returns an image as numpy array from figure
name = 'hussein'
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def task():
        for i in range(0,100000,1):
                print(name)
        return
from tensorflow.keras.models import Sequential, model_from_json
json_file = open('/home/justin/Desktop/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/justin/Desktop/model.h5")
one_image = 0
print("loaded model from disk")
t1 = Thread(target=task)
t1.start()

start = time.time()
X =[]
Y =[]

for i in range(0,100,1):
        print("ssss")
        X.append(1)
        Y.append(0)
plt.plot(X,Y)
print(plt.figure())
    
image = get_img_from_fig(plt.figure())                  
if (image is not None): 
        image = cv2.resize(image,(224,224))
        image = image/255
        image = np.array(image)
        image = np.expand_dims(image,axis=0)
        print("shape",image.shape)
        #x = loaded_model.predict([image])
        #print("predddddddddddD: ",x)
        
        #x = np.argmax(x)
t1.join()
print(time.time()-start)