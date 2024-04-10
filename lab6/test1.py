import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('grapefruit.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to smoothen the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding to create a binary image
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed algorithm
watershed_markers = cv2.watershed(image, markers)
watershed_boundaries_mask = np.zeros_like(image, dtype=np.uint8)
watershed_boundaries_mask[watershed_markers == -1] = [255, 0, 0]  # Mark watershed boundaries

# Display the image with watershed boundaries marked
plt.imshow(cv2.cvtColor(watershed_boundaries_mask, cv2.COLOR_BGR2RGB))
plt.title('Watershed Boundaries')
plt.show()
