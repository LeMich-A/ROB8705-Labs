import cv2
import numpy as np
# declare the path
path = r'/home/mich/Documents/Michel/Opencv/images/earth.jpg'
img = cv2.imread(path)

# show the original image
cv2.imshow('earthyyy', img)

# extract the RGB image channels
imgBlue = img[:, :, 0]  # blue channel
imgGreen = img[:, :, 1]  # green channel
imgRed = img[:, :, 2]  # red channel

# create blank images with zeros and fill each channel with the corresponding color
blue_channel = np.zeros_like(img)
blue_channel[:, :, 0] = imgBlue

green_channel = np.zeros_like(img)
green_channel[:, :, 1] = imgGreen

red_channel = np.zeros_like(img)
red_channel[:, :, 2] = imgRed

# display individual channels
cv2.imshow('Blue Channel', blue_channel)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Red Channel', red_channel)

imgResized = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Resized Image', imgResized)


imgHSV= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgHue = imgHSV[:, :, 0]
imgSat = imgHSV[:, :, 1]
imgVal= imgHSV[:, :, 2]


cv2.imshow('HSV Image', imgHSV)
cv2.imshow('Hue Channel', imgHue )
cv2.imshow('Saturation Channel', imgSat)
cv2.imshow('Value Channel', imgVal)

# Define the region to crop
start_row, end_row = 201, 400
start_col, end_col = 1301, 1500

# Crop the image
imgCrop = img[start_row:end_row, start_col:end_col]
cv2.imshow('Cropped Image', imgCrop)


# Rotate the cropped image by 30 degrees
rotation_angle = 30
rows, cols = img.shape[:2]

    # Create a rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)

    # Apply the rotation using the warpAffine function
imgRotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # Display the original, cropped, and rotated images
 
cv2.imshow('Rotated Image', imgRotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
