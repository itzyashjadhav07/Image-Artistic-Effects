from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm
import numpy as np

def get_pixel(image, i, j):
    width, height = image.size
    if i >= width or j >= height:
        return None
    return image.getpixel((i, j))


def color_average(image, i0, j0, i1, j1):
    red, green, blue, alpha = 0, 0, 0, 255
    width, height = image.size

    i_start = max(i0, 0)
    i_end = min(i1, width)
    j_start = max(j0, 0)
    j_end = min(j1, height)

    count = 0
    for i in range(i_start, i_end - 2, 2):
        for j in range(j_start, j_end - 2, 2):
            pixel = get_pixel(image, i, j)
            if pixel:
                r, g, b = pixel[:3]
                red += r
                green += g
                blue += b
                count += 1

    if count == 0:
        return 255, 255, 255, 255

    return int(red / count), int(green / count), int(blue / count), alpha


def convert_pointillize(image, radius=6):
    width, height = image.size
    new_image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(new_image)

    count = 0
    errors = [1, 0, 1, 1, 2, 3, 3, 1, 2, 1]

    for i in tqdm(range(0, width, radius + 3)):
        for j in range(0, height, radius + 3):
            color = color_average(image, i - radius, j - radius, i + radius, j + radius)
            eI = errors[count % len(errors)]
            count += 1
            eJ = errors[count % len(errors)]
            draw.ellipse((i - radius + eI, j - radius + eJ, i + radius + eI, j + radius + eJ), fill=color)

    return new_image


def apply_pointillism_effect(image_input, size=800):
    """
    Applies pointillism effect to the input image.

    Args:
        image_input (PIL.Image.Image or np.ndarray): The input image.
        size (int): Width to resize image to (maintains aspect ratio).

    Returns:
        PIL.Image.Image: Image with pointillism effect applied.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError("Input must be a PIL.Image.Image")

    # Resize while maintaining aspect ratio
    width, height = img.size
    new_height = int((size / width) * height)
    resized_img = img.resize((size, new_height))

    # Apply effect
    return convert_pointillize(resized_img)
