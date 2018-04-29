import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('pizza_bluescreen.jpg')
#plt.imshow(img)

#Image displayed as the background is red and this is mistake, but why?
#OpenCV read pixels [B,G,R] not [R,G,B] how can we solve this problem?
#By convert image from BGR to RGB

image_copy = np.copy(img)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
#plt.imshow(image_copy)

### Define the color threshold
## TODO: Define the color selection boundaries in RGB values
# play around with these values until you isolate the blue background

lower_blue = np.array([0,0,200])
upper_blue = np.array([250,250,255])

### Create a mask
# Define the masked area

mask = cv2.inRange(image_copy, lower_blue, upper_blue)
#plt.imshow(mask, cmap = 'gray')

# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)
masked_image[mask != 0] = [0,0,0]

#plt.imshow(masked_image)

### Mask and add a background image
# Load in a background image, and convert it to RGB 

background_image = cv2.imread('space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
#plt.imshow(background_image)


# Mask the cropped background so that the pizza area is blocked

background_image_copy = np.copy(background_image)
background_image_copy[mask == 0] = [0, 0, 0]
#plt.imshow(background_image_copy)
# Mask the cropped background so that the pizza area is blocked
#crop_masked_image[mask == 0] = [0, 0, 0]


complete_image = masked_image + background_image
plt.imshow(complete_image)
