import cv2
import os
import streamlit as st
import argparse

from blur_number_plate_in_image import blur_number_plates, load_model


def blur_number_plates_in_video(video_path, output_path, model, use_progress=True):

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if use_progress:
        progress_bar = st.progress(0)
    frame_count = 0
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Save the output video
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (frame_width, frame_height))
    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Blur the detected number plate
            processed_frame = blur_number_plates(frame, model)
            out.write(processed_frame)
            frame_count += 1

            progress_percentage = frame_count / max_frames
            if use_progress:
                progress_bar.progress(progress_percentage)

    finally:
        print("Releasing Media")
        cap.release()
        out.release()


if __name__ == "__main__":

    # Load the model
    model = load_model()
    model.eval()

    parser = argparse.ArgumentParser(description="Blurring Video")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str,
                        default="./blurred_video.mp4")
    args = parser.parse_args()

    path, filename = os.path.split(args.output_path)
    os.makedirs(path, exist_ok=True)

    blur_number_plates_in_video(
        args.input_path, args.output_path, model, False)
