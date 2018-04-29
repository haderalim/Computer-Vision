## Which area has the lowest value for red? What about for blue?

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img = mpimg.imread('wa_state_highway.jpg')


#Visualize the levels of each color channel. Pay close attention to the traffic signs!
# Isolate RGB channels

r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('R channel')
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(b, cmap='gray')

## Which area has the lowest value for red? What about for blue?
min_value_for_red = np.amin(r)


min_value_for_blue = np.amin(b)


print 'Area has the lowest value for red:'

min_value_for_red = np.amin(r)
for (x,y), pixel in np.ndenumerate(r):
    if( pixel == min_value_for_red ):
        print (x,y)
