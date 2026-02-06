# Image Segmentation Project

Image segmentation with DeepLabV3 and MediaPipe.

## Description

This project uses the **DeepLabV3** model to segment objects in images. DeepLabV3 is a CNN-based architecture. The model is trained on the PASCAL VOC dataset, which includes 21 classes (20 object classes + background). It classifies each pixel into one of these classes, allowing for precise segmentation of objects in images.

The project processes images from a `dataset` folder organized by categories and generates composite results in the `output` folder.

## Features

- **White/Gray Mask**: Displays detected objects in white on a gray background
- **Contours**: Extracts and displays object contours
- **Blurred Background**: Creates a selective blur effect on the background
- **Composite Image**: Combines all 4 results side by side

## Project Structure

```
.
├── test_model.py                   # Main script
├── deeplabv3.tflite                # Segmentation model
├── dataset/                        # Input folder
└── output/                         # Output folder (results)
```

## Installation

```bash
pip install opencv-python numpy mediapipe
```

## Usage

```bash
python main_refactored.py
```

The script processes all images from the `dataset/` folder and saves results to `output/`.

## Parameters

Configurable in the code:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DESIRED_HEIGHT` | 480 | Target display height |
| `DESIRED_WIDTH` | 480 | Target display width |
| `MASK_THRESHOLD` | 0.2 | White/gray mask threshold |
| `CONTOUR_THRESHOLD` | 0.2 | Contour extraction threshold |
| `BLUR_THRESHOLD` | 0.1 | Background blur threshold |
| `BLUR_KERNEL` | (55, 55) | Gaussian kernel size |

## Supported Formats

- JPG
- JPEG
- PNG

## Dataset

Source: [Images Dataset on Kaggle](https://www.kaggle.com/datasets/pavansanagapati/images-dataset)

**Note**: Only the first 9 images from each folder (cats, dogs, flowers, horses, human) were kept for this project.

## Output example

The output image will be a composite of the original image, white/gray mask, contours, and blurred background, displayed side by side.

![Example Output](output/cats/cat.1.png)
