import streamlit as st
import numpy as np
from PIL import Image
import io
from utils import kmeans

st.title("Image quantizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", 'tif', 'tiff'])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image = np.array(pil_image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    user_input = st.text_input("How many colors do you want in the image?", "16")

    if user_input.isdigit():
        is_valid = True
    else:
        is_valid = False

    search = st.checkbox("Search for optimal colors", value=False)

    n = 10 if search else 1
    K = int(user_input)

    formats = ["PNG", "JPEG"]
    if K < 256:
        formats.append("GIF")

    st.session_state.format = st.selectbox("In which format do you want to download the result ?", formats)

    if st.button("Submit", disabled=not is_valid):
        st.session_state["submitted"] = True

    if st.session_state.get("submitted"):
        with st.spinner("Processing... Please wait ⏳"):
            results = []
            for _ in range(n):
                centroids, idx, J = kmeans(image.reshape(-1, 3), K)
                results.append((centroids, idx, J))
            centroids, idx, J = min(results, key=lambda x: x[2])

        new_image = centroids[idx].reshape(image.shape).astype(np.uint8)

        st.image(new_image, caption="Quantized Image", use_container_width=True)
        st.write(f"Distorsion: {J:.2f}")

        new_image = Image.fromarray(new_image)
        image_bytes = io.BytesIO()

        new_image.save(image_bytes, format=st.session_state.format)
        image_bytes.seek(0)

        st.download_button(label="Click here to download the image", 
                           file_name=f"quantized_image.{st.session_state.format.lower()}", 
                           data=image_bytes, 
                           key="download")
        
    