#Dilation and Erosion 

import cv2 
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------Dilation------------------------------
# Reads in a binary image
img = cv2.imread('j.png',0)

# Create a 5x5 kernel of ones
Kernel = np.ones((5,5), np.uint8)


'''
To dilate an image in OpenCV, you can use the dilate function and three inputs: 
an original binary image, a kernel that determines the size of the dilation (None will 
result in a default size), and a number of iterations to perform the dilation (typically = 1).
In the below example, we have a 5x5 kernel of ones, which move over an image, like a filter, 
and turn a pixel white if any of its surrounding pixels are white in a 5x5 window! We’ll 
use a simple image of the cursive letter “j” as an example.
'''

dilation = cv2.dilate(img, Kernel, iterations = 1)
plt.imshow(dilation, cmap = 'gray')


#-----------------------------------Erosion--------------------------------
erosion = cv2.erode(img, Kernel, iterations = 1)
plt.imshow(erosion, cmap = 'gray')

#----------------------------------Opening------------------------------

'''
As mentioned, above, these operations are often combined for desired results! One such combination
 is called opening, which is erosion followed by dilation. This is useful in noise reduction
 in which erosion first gets rid of noise (and shrinks the object) then dilation enlarges the
 object again, but the noise will have disappeared from the previous erosion!

To implement this in OpenCV, we use the function morphologyEx with our original image,
 the operation we want to perform, and our kernel passed in.
 '''
 
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel)
plt.imshow(opening, cmap = 'gray')

#----------------------------------Closing------------------------------

'''
Closing is the reverse combination of opening; it’s dilation followed by erosion,
 which is useful in closing small holes or dark areas within an object.

Closing is reverse of Opening, Dilation followed by Erosion. It is useful in
 closing small holes inside the foreground objects, or small black points on the object.
'''

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel)
plt.imshow(closing, cmap = 'gray')
