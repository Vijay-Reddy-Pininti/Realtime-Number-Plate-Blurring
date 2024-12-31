import cv2

from blur_number_plate_in_image import blur_number_plates, load_model


def blur_number_plates_in_webcam(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Blur the detected number plate
            processed_frame = blur_number_plates(frame, model)
            cv2.imshow("Processed Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Releasing Media")
        cap.release()


if __name__ == "__main__":
    # Load the model
    model = load_model()
    model.eval()
    blur_number_plates_in_webcam(model)
