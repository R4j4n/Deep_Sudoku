from utils import *
from backtracking import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np 
from scipy import ndimage
import cv2


################# INPUT PATH OF YOUR IMAGE###############################
path = 'D:\\Projects\\SODUKU SOLVER\\test_image\\1.jpg'
result = find_result(path)
cv2.imshow('result',result)
cv2.waitKey(0)
