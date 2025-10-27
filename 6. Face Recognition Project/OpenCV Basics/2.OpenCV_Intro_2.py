# Reading file in OpenCV

import cv2

img = cv2.imread('./dog_img.png')
gray = cv2.imread('./dog_img.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Dog Image', img)
cv2.imshow('Gray Dog Image', gray)

cv2.waitKey(5000)                            #Time to wait in milliseconds
cv2.destroyAllWindows()                      #close all the window 