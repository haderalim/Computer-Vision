import numpy as np
import matplotlib.pyplot as plt
import cv2

### -----------------------------Import resources and display image----------------------------

# Read in the image
## TODO: Check out the images directory to see other images you can work with
# And select one!
image = cv2.imread('monarch.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)


### -----------------------------### Prepare data for k-means----------------------------
# Reshape image into a 2D array of pixels and 3 color values (RGB)
#Image should be m by 3 in dimension, where m is number of pixels and three is number of color channels.
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)


### ----------------------------Implement k-means clustering----------------------------
# define stopping criteria
# you can change the number of max iterations for faster convergence!
#100 is a number of iterations 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

## TODO: Select a value for k
# then perform k-means clustering
k = 3
#None:Then any labels we want and None in this case
#10: Number of attempts choose around center point
#The criteria tells the algorithm when to stop
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
#Then return the shape to the shape of the original image
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)

## TODO: Visualize one segment, try to find which is the leaves, background, etc!

plt.imshow(labels_reshape==0, cmap='gray')

# mask an image segment by cluster

cluster = 0 # the first cluster

masked_image = np.copy(image)
# turn the mask green!
masked_image[labels_reshape == cluster] = [0, 255, 0]

plt.imshow(masked_image)





