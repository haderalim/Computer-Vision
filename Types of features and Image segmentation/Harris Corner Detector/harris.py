# Harris Corner Detection

###------------------------------ Import resources and display image---------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Read in the image
#img = cv2.imread('waffle.jpg')
img = cv2.imread('chessboard.png')

# Make a copy of the image
img_copy = np.copy(img)


# Change color to RGB (from BGR)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)


###--------------------------------Detect corners---------------------------------

# Convert to grayscale
gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

#Harris corner detector algorithm use float32 
gray = np.float32(gray)

# Detect corners 
#cornerHarris method takes 
#parameter1: gray scale float values 
#parameter2: size of neighborhood to look at when identifying potential corners, 2 means 2x2 pxel square 
#parameter3: Size of sobel operator 3 which is typical kernel size
#parameter4: A constant Value that helps determine which points are considered corners

dst = cv2.cornerHarris(gray, 2, 3, 0.04 )
plt.imshow(dst, cmap = 'gray')


# Dilate corner image to enhance corner points
#Dilation process:- enlarge bright regions or regions in the foreground like corners, so that be 
#able to see them better.
#dst: The image should have the corners marked as bright points and non corners as darker pixels.
dst = cv2.dilate(dst, None)
plt.imshow(dst, cmap = 'gray')

#To select the strongest corners, I will define a threshold value for them to pass.
## TODO: Define a threshold for extracting strong corners
# This value vary depending on the image and how many corners you want to detect
# Try changing this free parameter, 0.1, to be larger or smaller ans see what happens

thresh = 0.1*dst.max()

#----------------------- Create an image copy to draw corners on image---------------
corner_image = np.copy(img_copy)

# Iterate through all the corners and draw them on the image (if they pass the threshold)

for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if(dst[j,i] > thresh):
            # image, center pt, radius, color, thickness
            cv2.circle(corner_image, (i,j), 1, (0,255,0), 1 )
            
            
plt.imshow(corner_image)
