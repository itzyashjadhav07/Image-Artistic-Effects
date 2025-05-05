from PIL import Image
from tqdm import tqdm

def get_max(value):
    return min(255, int(value))

def get_sepia_pixel(r, g, b, alpha=255):
    tRed = get_max(0.759 * r + 0.398 * g + 0.194 * b)
    tGreen = get_max(0.676 * r + 0.354 * g + 0.173 * b)
    tBlue = get_max(0.524 * r + 0.277 * g + 0.136 * b)
    return (tRed, tGreen, tBlue, alpha)

def convert_sepia(image):
    width, height = image.size
    sepia_image = Image.new("RGB", (width, height), "white")
    pixels = sepia_image.load()
    
    for i in tqdm(range(width), desc="Applying Sepia Effect"):
        for j in range(height):
            r, g, b = image.getpixel((i, j))[:3]
            pixels[i, j] = get_sepia_pixel(r, g, b)
    
    return sepia_image

def apply_sepia_effect(image_input, size=800):
    """
    Applies sepia tone to the input image.

    Args:
        image_input (PIL.Image.Image): The input image.
        size (int): Width to resize the image to.

    Returns:
        PIL.Image.Image: The sepia-processed image.
    """
    if not isinstance(image_input, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image object")
    
    # Resize image
    width, height = image_input.size
    new_height = int((size / width) * height)
    resized_img = image_input.resize((size, new_height))
    
    # Apply sepia effect
    return convert_sepia(resized_img)
