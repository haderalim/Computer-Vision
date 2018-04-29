import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read in the image
image = cv2.imread('water_balloons.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

#plt.imshow(image)

##Plot color channels
# RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Red')
ax1.imshow(r, cmap='gray')

ax2.set_title('Green')
ax2.imshow(g, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(b, cmap='gray')


# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#plt.imshow(hsv)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Hue')
ax1.imshow(h, cmap='gray')

ax2.set_title('Saturation')
ax2.imshow(s, cmap='gray')

ax3.set_title('Value')
ax3.imshow(v, cmap='gray')



### Define pink and hue selection thresholds

# Define our color selection criteria in HSV values
lower_hue = np.array([160,0,0]) 
upper_hue = np.array([180,255,255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180,0,100]) 
upper_pink = np.array([255,255,230])

#Apply mask to get Binary Image
mask_rgb = cv2.inRange(image, lower_pink, upper_pink )
plt.imshow(mask_rgb, cmap = 'gray')
#cv2.imshow('Mask of RGB Image', mask_rgb)

mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)
plt.imshow(mask_hsv, cmap = 'gray')

#Aply mask to get Color Image
#Applying on RGB Image
masked_image = np.copy(image)
masked_image[mask_rgb == 0] = [0,0,0]
plt.imshow(masked_image)

#Applying on HSV Image
masked_hsv = np.copy(hsv)
masked_hsv[mask_hsv == 0] = [0,0,0]
plt.imshow(masked_hsv)
