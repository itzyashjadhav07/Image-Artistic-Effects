import cv2  # Image processing
import numpy as np  # Matrix operations
from time import sleep  # Simulate time delay for visualizing progress
from tqdm import tqdm  # Progress bar
import argparse
import os  # For file path operations

def comic(img):
    sleep(0.1)
    with tqdm(total=100, desc="Applying Comic Effect") as pbar:
        # Convert to grayscale for edge detection
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pbar.update(10)

        # Apply Canny edge detection
        edgesOnly = cv2.Canny(grayImg, 50, 150)
        sleep(0.1)
        pbar.update(20)

        # Smooth the image with bilateral filter
        color = cv2.bilateralFilter(img, 9, 300, 300)
        sleep(0.1)
        pbar.update(30)

        # Invert and convert edges to 3-channel
        edgesOnlyInv = cv2.bitwise_not(edgesOnly)
        edgesOnlyInv = cv2.cvtColor(edgesOnlyInv, cv2.COLOR_GRAY2BGR)

        # Blend edges into color image
        cartoon = cv2.addWeighted(color, 0.9, edgesOnlyInv, 0.2, 0)
        sleep(0.1)
        pbar.update(20)

        # Apply sharpening filter
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        cartoon = cv2.filter2D(cartoon, -1, kernel)
        sleep(0.1)
        pbar.update(20)

        pbar.update(10)  # Final touch

    return cartoon

# Argument parser for input and output image paths
ap = argparse.ArgumentParser(description="Apply comic cartoon effect to an image.")
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-o", "--output", default="assets/comic_cartoon_effect.jpg", help="Path to save the output image")
args = vars(ap.parse_args())

# Load the image
img = cv2.imread(args["image"])

# Validate image load
if img is None:
    print(f"Error: Unable to load the image from '{args['image']}'. Check the path and format.")
    exit()

# Ensure output directory exists
os.makedirs(os.path.dirname(args["output"]), exist_ok=True)

print("Wait, work is in progress...")

# Apply comic effect
res_img = comic(img)

# Save the result
cv2.imwrite(args["output"], res_img)
print(f"Done! Your result has been saved to: {args['output']}")
