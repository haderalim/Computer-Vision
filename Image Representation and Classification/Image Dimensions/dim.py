import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg

#Read image
#image size (490, 655, 3) in uint8 datatype
img = mpimg.imread('car.jpeg')


#To print image dimensions
print 'Image dimensions  = ', img.shape
