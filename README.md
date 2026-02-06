# Image Segmentation Project

Image segmentation with DeepLabV3 and MediaPipe.

## Description

This project uses the **DeepLabV3** model to segment objects in images. It processes images from a `dataset` folder organized by categories and generates composite results in the `output` folder.

## Features

- **White/Gray Mask**: Displays detected objects in white on a gray background
- **Contours**: Extracts and displays object contours
- **Blurred Background**: Creates a selective blur effect on the background
- **Composite Image**: Combines all 4 results side by side

## Project Structure

```
.
├── test_model.py          # Main script
├── main.py                      # Alternative version
├── image_segmentation.ipynb     # Jupyter notebook
├── deeplabv3.tflite             # Segmentation model
├── dataset/                     # Input folder (categories)
│   ├── cats/
│   ├── dogs/
│   ├── flowers/
│   ├── horses/
│   └── human/
└── output/                      # Output folder (results)
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
