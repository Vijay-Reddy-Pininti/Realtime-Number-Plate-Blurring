import torch
import cv2
import argparse


_model = None


def load_model():

    global _model

    if _model is not None:
        return _model  # Return the cached model if it's already loaded

    model_path = './exp_200/exp/weights/best.pt'
    try:
        _model = torch.hub.load('WongKinYiu/yolov9',
                                'custom', model_path, force_reload=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        _model = None

    return _model


def blur_number_plates(image, model):

    if image is None:
        print("Image not found")

    # Detect number plates
    results = model(image)

    # Process detections
    for det in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        if conf > 0.5:  # confidence threshold
            # Extract the detected number plate region
            plate_region = image[y1:y2, x1:x2]

            # Apply Gaussian blur on detected number plate
            blurred_plate = cv2.GaussianBlur(plate_region, (99, 99), 0)

            # Place the blurred region back into the image
            image[y1:y2, x1:x2] = blurred_plate

    return image


if __name__ == "__main__":

    # Load the model
    model = load_model()
    model.eval()

    parser = argparse.ArgumentParser(description="Blurring Image")
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.img_path)
    blurred_image = blur_number_plates(image, model)

    cv2.imwrite("blurred_image.png", blurred_image)
