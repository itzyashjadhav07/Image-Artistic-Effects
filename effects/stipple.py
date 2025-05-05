import argparse
import random
from PIL import Image
from tqdm import tqdm

def stippler(img, width, height):
    imgNew = Image.new('L', (width, height))
    for x in tqdm(range(width)):
        for y in range(height):
            gray_value = img.getpixel((x, y))
            randNum = random.randint(0, 255)
            if randNum >= gray_value:
                imgNew.putpixel((x, y), 0)  # Black dot
            else:
                imgNew.putpixel((x, y), 255)  # White background
    return imgNew

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(ap.parse_args())

# Open and convert image
img = Image.open(args['image']).convert('L')  # Convert to grayscale
width, height = img.size

# Apply stippling effect
stip_img = stippler(img, width, height)

# Save result
stip_img.save('assets/Stippled.png')
