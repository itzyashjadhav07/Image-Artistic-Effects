import cv2
import numpy as np
from PIL import Image

def apply_pencil_sketch_effect(image_input, size=800):
    """
    Applies pencil sketch (black & white and color) effects to an image.

    Args:
        image_input (PIL.Image.Image or np.ndarray): Input image.
        size (int): Target width (height auto-adjusted to maintain aspect ratio).

    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image]: (Black & white sketch, color sketch)
    """
    # Convert PIL to OpenCV if needed
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("Input must be a PIL.Image or numpy.ndarray")

    if image is None:
        raise ValueError("Invalid image input (None)")

    # Resize image maintaining aspect ratio
    height, width = image.shape[:2]
    new_height = int((size / width) * height)
    resized_image = cv2.resize(image, (size, new_height))

    # Apply pencil sketch effect
    sketch_bw, sketch_color = cv2.pencilSketch(
        resized_image, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )

    # Convert to RGB for PIL compatibility
    sketch_bw = cv2.cvtColor(sketch_bw, cv2.COLOR_GRAY2RGB)
    sketch_color = cv2.cvtColor(sketch_color, cv2.COLOR_BGR2RGB)

    return Image.fromarray(sketch_bw), Image.fromarray(sketch_color)
