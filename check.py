from PIL import Image
import os

# Define the directory containing the images
image_directory = './samples'

# Loop through each file in the directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Construct the full file path
        file_path = os.path.join(image_directory, filename)
        # Open the image
        with Image.open(file_path) as img:
            # Rotate the image 270 degrees
            rotated_img = img.rotate(270, expand=True)
            # Save the rotated image back to the same file
            rotated_img.save(file_path)

# Print a message when done
print("All images in the directory have been rotated and saved.")
