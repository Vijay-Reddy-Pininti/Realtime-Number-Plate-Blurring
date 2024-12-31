import streamlit as st
import cv2
import numpy as np

from blur_number_plate_in_image import blur_number_plates, load_model
from blur_number_plate_in_video import blur_number_plates_in_video


@st.cache_resource
def load_cached_model():
    model = load_model()
    if model is not None:
        model.eval()
    return model


model = load_cached_model()

# Header
st.title("Realtime Number Plate Blurring")

# Accepted Input types
st.markdown("""Upload an image or a video and click on **Submit** to apply a blur effect. 
You will be able to download the blurred file once the process is complete.""")

# File uploader for images and videos
with st.form("file_upload_form"):
    uploaded_file = st.file_uploader("Upload an image or video", type=[
                                     "jpg", "png", "mp4", "jpeg", "mpeg4"], accept_multiple_files=False)
    submit_button = st.form_submit_button("Submit")

if submit_button and uploaded_file is not None:
    file_type = uploaded_file.type
    if "image" in file_type:
        try:
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, caption="Original Image",
                     channels="BGR", use_container_width=True)

            blurred_image = blur_number_plates(image, model)
            st.image(blurred_image, caption="Blurred Image",
                     channels="BGR", use_container_width=True)

            # Download button for image
            cv2.imwrite("blurred_image.png", blurred_image)
            with open("blurred_image.png", "rb") as file:
                btn = st.download_button(
                    label="Download Blurred Image",
                    data=file,
                    file_name="blurred_image.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error processing image: {e}")
    elif "video" in file_type:  # Check if the file is a video
        try:
            video_path = f"uploaded_video.mp4"
            with open(video_path, "wb") as video_file:
                video_file.write(uploaded_file.getbuffer())

            out_path = "blurred_video.mp4"

            blur_number_plates_in_video(video_path, out_path, model)

            st.video(out_path)
            with open(out_path, "rb") as file:
                st.download_button(
                    label="Download Blurred Video",
                    data=file,
                    file_name="blurred_video.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Error processing video: {e}")
    else:
        st.error("Unsupported file format. Please upload a .jpg, .png, or .mp4 file.")
