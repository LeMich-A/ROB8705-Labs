import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import random

# declare the path
path = r'/home/mich/Documents/Michel/Opencv/images/saphira.jpg'
img = cv2.imread(path)

cv2.imshow("image saphira ", img)


#create the kernel mean
kernelMean =  np.ones((31,31),np.float32)/ 961 # create array of size 31 and for kernel mean we use ones who take the ((size,size), np.float32) create the 2 D array filed with 1 as 32 bit floating point numbers , the last part is to normilizing the array for an overall brightness

# apply it to the image now 
imgMean = cv2.filter2D(img,-1,kernelMean)# is the depth of the image to be same as original image (default)

#create gUSSIAN FILTER
kernelGaussian = cv2.getGaussianKernel(31, 5)# create a 1D array gaussian 
gauss = np.outer(kernelGaussian ,kernelGaussian )# convert it to a 2D array


# using the matplot we will create an surface plot of the gaussian filter 
i = 31 # define the size of parameter of square kernal
fig = plt.figure()# create the plot
ax = fig.add_subplot(1,2,1, projection='3d')# add subplot to the figure plot created saying its a 1x2 grid
x = np.arange(0, i, 1)#create the x axis
y = np.arange(0, i, 1)#create the y axis
X, Y = np.meshgrid(x, y) #create 2D array of our grid
Z = gauss.flatten()#create the Z axis even if it flats into the z
ax.plot_surface(X, Y, gauss, cmap='viridis')
plt.show()

#add gaussian blur 
Gaussblur = cv2.GaussianBlur(img,(31,31),5)

# showing in one windows
Fenetre2 = cv2.hconcat([img,imgMean, Gaussblur])# add them in one windows all three image for better comparaison
cv2.imshow('Original + imgMean + Gaussian Blur Filter', Fenetre2) #same way we use all imshow

# add noise can only add it to random pixel or pixel of your choice , we will go with randomn
img_noise = np.float32(img)

# Set the standard deviation for the Gaussian noise
std_dev = 25

# Generate Gaussian noise using cv2.randn
noise = np.zeros_like(img)
cv2.randn(noise, 0, std_dev)# noise to randomn point

# Add the generated noise to the image
noisy_image = cv2.add(img, noise)

# Clip the pixel values to be in the valid range [0, 255]
noisy_image = np.clip(noisy_image, 0, 255)

# Convert the result back to uint8
noisy_image = np.uint8(noisy_image)

kernelMean_2 =  np.ones((3,3),np.float32)/ 9
# create imgmean to image
imgMean2 = cv2.filter2D(noisy_image,-1,kernelMean_2)

#add the blur to image
Gaussblur2 = cv2.GaussianBlur(noisy_image,(3,3),0.5)

medianBlur= cv2.medianBlur(noisy_image,3)

Fenetre3= cv2.hconcat([noisy_image, Gaussblur2 ,imgMean2, medianBlur])
cv2.imshow('seasoned pic + Averaging Filter + Gausiann Filter + Median Blur', Fenetre3)

cv2.waitKey(0)
cv2.destroyAllWindows()
