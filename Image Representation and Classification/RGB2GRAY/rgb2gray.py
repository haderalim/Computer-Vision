import cv2

#RGB mat for storing RGB image
RGB = cv2.imread('tom.jpg')

#Gray mat for storing result of conversion from RGB to Gray
Gray = cv2.cvtColor(RGB,cv2.COLOR_BGR2GRAY)

#to display an image in window
cv2.imshow('Gray Image',Gray)

#waits for a pressed key
k = cv2.waitKey(0)

#Esc key to stop
if k == 27:
	cv2.destroyAllWindows()
