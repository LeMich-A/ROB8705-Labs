import numpy as np
import cv2 


img = cv2.imread("grapefruit.jpg")



img_kernel = np.array([[1,0,0,0,1],[0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0],[1,0,0,0,1]]) 
Img_kernel = img_kernel * (1/9)

flt_img = cv2.filter2D(img,-1,Img_kernel)

cv2.imshow('Diagonal_Filter', flt_img)

cv2.waitKey(0) 
cv2.destroyAllWindows()