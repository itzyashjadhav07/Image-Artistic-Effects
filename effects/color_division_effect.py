import os
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import numpy as np
import argparse
import cv2

# Create assets folder if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

def color_divisionK3(image):
    with tqdm(total=100, desc="Progress 1/2: ") as pbar:
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pbar.update(10)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=3)
        pbar.update(30)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        pbar.update(30)
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        res = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        pbar.update(30)
    return res

def color_divisionK6(image):
    with tqdm(total=100, desc="Progress 2/2: ") as pbar:
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pbar.update(10)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=6)
        pbar.update(30)
        labels = clt.fit_predict(image)
        pbar.update(20)
        res = clt.cluster_centers_.astype("uint8")[labels]
        pbar.update(10)
        res = res.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        res = cv2.cvtColor(res, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        pbar.update(30)
    return res

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Reading the image
img = cv2.imread(args["image"])

# Check if image was loaded properly
if img is None:
    print("Error: Unable to load the image. Please check the file path and format.")
    exit(1)

# Apply the color division effects
result1 = color_divisionK3(img)
result2 = color_divisionK6(img)

# Save the results
cv2.imwrite("assets/colorDivisionK3.jpg", result1)
cv2.imwrite("assets/colorDivisionK6.jpg", result2)
print("Your results are ready!")
