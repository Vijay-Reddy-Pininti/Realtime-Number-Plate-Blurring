# Realtime Number Plate Blurring

This project focuses on capturing and blurring vehicle number plates in real time to address privacy concerns. For instance, a vacation vlogger recording their environment may inadvertently capture private license plate information. This tool helps blur license plates in real-time, reducing the need for post-production editing.

## Features
- Detects vehicle license plates using YOLOv9.
- Applies Gaussian blur specifically to license plates.
- Configured for custom datasets to achieve optimal performance.
- Capable of processing images and videos in real time.

## State of the Art
While license plate recognition is a well-researched area, existing solutions focus primarily on extracting textual information rather than blurring plates. This project uniquely addresses privacy concerns by blurring license plates in real time.

## Inputs and Outputs
- **Inputs**: Images or videos containing vehicle license plates.
- **Outputs**: Images or videos with blurred license plates.

## Key Contributions
1. Configured YOLOv9 for custom dataset training.
2. Trained the model to detect license plates with high confidence.
3. Developed a real-time blurring mechanism using Gaussian blur.
4. Conducted extensive experiments to optimize model performance.

## Approach
1. **YOLOv9**: Used for detecting vehicle license plates with high speed and accuracy.
2. **Gaussian Blur**: Applied to the detected regions to ensure privacy by blurring license plates.

### Pros
- High accuracy and speed for real-time applications.
- Flexible and customizable for various datasets and environments.

### Cons
- Requires high computational resources for optimal performance.
- May struggle with small or distant license plates in images.

## Datasets Used
[Licence Plate Detection Computer Vision dataset](https://universe.roboflow.com/mashinelearning/licence-plate-detection-wcfzj/dataset/9)

## Results
- The model achieved a mean average precision (mAP) of **0.78** after 200 epochs.
- Successfully detected and blurred license plates with confidence levels ranging from **50%** to **86%**.

## Requirements
- **Hardware**: NVIDIA T4 GPU or equivalent.
- **Software**:
  - Python 3.8
  - OpenCV
  - PyTorch
  - NumPy
  - Matplotlib

## Future Applications
- Integrating with CCTV systems for privacy preservation in public places.
- Complying with privacy regulations for publicly accessible video feeds.
- Anonymizing traffic footage for investigations.

## Lessons Learned
- The importance of a high-quality, diverse dataset for robust model performance.
- The strengths of YOLOv9 in real-time object detection.

## How to Use

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Vijay-Reddy-Pininti/Realtime-Number-Plate-Blurring.git
   cd Realtime-Number-Plate-Blurring
   ```
   
2. **Download the dataset** </br>
- Download the dataset from this link: [Licence Plate Detection Computer Vision dataset](https://universe.roboflow.com/mashinelearning/licence-plate-detection-wcfzj/dataset/9).
- Extract the downloaded file and copy the **test, train** and **valid** folders into the **Realtime-Number-Plate-Blurring** folder.


3. **Download the pre-trained weights**

- Download the pre-trained weights (yolov9-e.pt) from this link: [Weights](https://github.com/WongKinYiu/yolov9/releases/tag/v0.1).
- Copy the weights into the **Realtime-Number-Plate-Blurring** folder.

4. **Training the model**

- Run the file **train_dataset.ipynb**.
- Adjust the epochs if required for your dataset.

5. **Blur License Plates in Image**

```bash
python blur_number_plate_in_image.py --img_path <your_image>
```

6. **Blur License Plates in Video**

```bash
python blur_number_plate_in_video.py --input_path <video_to_blur> --output_path <output_destination>
```

7. **Blur License Plates in Realtime**

```bash
python blur_number_plate_in_webcam.py
Press "ctrl + c" once you are done.
```
    

## Video Demonstration
Watch the video demonstration of the project on [YouTube](https://youtu.be/i53-ycDQwh8?si=Yk9nWd7VKUjslxx1).

## References
- [YOLOv9 GitHub Repository](https://github.com/WongKinYiu/yolov9)
- [Gaussian Blur Documentation](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)

## Contact
[Vijay Reddy Pininti](https://www.linkedin.com/in/vijay-reddy-pininti/)

---

Feel free to contribute, raise issues, or share your feedback!
