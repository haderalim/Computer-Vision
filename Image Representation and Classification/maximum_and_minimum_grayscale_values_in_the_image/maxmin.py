import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg

#Read image
#image size (490, 655, 3) in uint8 datatype
img = mpimg.imread('car.jpeg')


#To print image dimensions
print 'Image dimensions  = ', img.shape
#Convert image from RGB to GRAY
#image size  will be (490, 655) in uint8 datatype
grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(grayimg, cmap='gray')

#Find the maximum and minimum grayscale values in this image
max_value = np.amax(grayimg)
min_value = np.amin(grayimg)

print 'maximum value: ',max_value
print 'minimum value: ',min_value
