import os
import cv2

def process_image(image_path):# create function to run in the script , grayscale and laplacian 
    
    img = cv2.imread(image_path)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter
    laplacian_image = cv2.Laplacian(img, cv2.CV_64F)

    # Update image names
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    grayscale_name = f"{base_name}_grayscale.jpg"
    laplacian_name = f"{base_name}_laplacian.jpg"

    # Save processed images
    cv2.imwrite(grayscale_name, grayscale_image)
    cv2.imwrite(laplacian_name, laplacian_image)

def processimgdic(directory_path):
    
    # List all files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        process_image(image_path)

if __name__ == "__main__":
   
    input_directory = "/home/mich/Documents/Michel/Opencv/images"

    # Process images in the specified directory
    processimgdic(input_directory)
    
   




