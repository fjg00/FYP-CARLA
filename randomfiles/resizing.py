from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import time
import matplotlib.pyplot as plt


image1 = cv2.imread('/home/justin/Downloads/notraffic.jpg')
print(image1.shape)
image1 = cv2.resize(image1,(225,224))

print(image1.shape)

cv2.waitKey(0)
cv2.imwrite("/home/justin/Downloads/no_traffic.jpg",image1)