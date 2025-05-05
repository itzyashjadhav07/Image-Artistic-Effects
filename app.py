import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Import each effect function from the effects folder
from effects.anime_effect import animefy
from effects.cartoon_effects import cartoonize
from effects.comic_cartoon_effect import comic
from effects.color_division_effect import color_divisionK3
from effects.low_poly import apply_low_poly_effect
from effects.motion_blur import motionBlur
from effects.negative import apply_negative_effect
from effects.oil_painting import apply_oil_painting_effect
from effects.pencil_sketch import apply_pencil_sketch_effect
from effects.pointilism import apply_pointillism_effect
from effects.sepia_effect import apply_sepia_effect
from effects.stipple import stippler


# Streamlit app title
st.title("Image Artistic Effects")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Resize input image to a smaller size for better performance
    target_width = 480
    height, width = img.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    img = cv2.resize(img, (target_width, new_height))

    # Choose the effect
    effect = st.selectbox("Choose the artistic effect", [
        "Anime Effect", 
        "Cartoon Effect", 
        "Comic Cartoon Effect", 
        "Color Division Effect", 
        "Low Poly Effect", 
        "Motion Blur", 
        "Negative Effect", 
        "Oil Painting", 
        "Pencil Sketch", 
        "Pointilism Effect", 
        "Sepia Effect",  
        "Stipple Effect", 
    ])

    # Apply selected effect
    if effect == "Anime Effect":
        processed_img = animefy(img)
        st.image(processed_img, caption="Anime Effect", use_column_width=True)

    elif effect == "Cartoon Effect":
        processed_img = cartoonize(img)
        st.image(processed_img, caption="Cartoon Effect", use_column_width=True)

    elif effect == "Comic Cartoon Effect":
        processed_img = comic(img)
        st.image(processed_img, caption="Comic Cartoon Effect", use_column_width=True)

    elif effect == "Color Division Effect":
        processed_img = color_divisionK3(img)
        st.image(processed_img, caption="Color Division Effect", use_column_width=True)

    elif effect == "Low Poly Effect":
        processed_img = apply_low_poly_effect(img)
        st.image(processed_img, caption="Low Poly Effect", use_column_width=True)

    elif effect == "Motion Blur":
        processed_img = motionBlur(img)
        st.image(processed_img, caption="Motion Blur Effect", use_column_width=True)

    elif effect == "Negative Effect":
        processed_img = apply_negative_effect(img)
        st.image(processed_img, caption="Negative Effect", use_column_width=True)

    elif effect == "Oil Painting":
        processed_img = apply_oil_painting_effect(img)
        st.image(processed_img, caption="Oil Painting Effect", use_column_width=True)

    elif effect == "Pencil Sketch":
        processed_img = apply_pencil_sketch_effect(img)
        st.image(processed_img, caption="Pencil Sketch Effect", use_column_width=True)

    elif effect == "Pointilism Effect":
        processed_img = apply_pointillism_effect(img)
        st.image(processed_img, caption="Pointilism Effect", use_column_width=True)

    elif effect == "Sepia Effect":
        processed_img = apply_sepia_effect(img)
        st.image(processed_img, caption="Sepia Effect", use_column_width=True)



    elif effect == "Stipple Effect":
        processed_img = stippler(img)
        st.image(processed_img, caption="Stipple Effect", use_column_width=True)



    # Allow users to download the processed image
    download_button = st.download_button(
        label="Download Processed Image",
        data=cv2.imencode('.jpg', processed_img)[1].tobytes(),
        file_name="transformed_image.jpg",
        mime="image/jpeg"
    )
