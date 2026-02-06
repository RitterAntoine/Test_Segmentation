import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
BG_COLOR = (192, 192, 192)
MASK_COLOR = (255, 255, 255)
MODEL_PATH = 'deeplabv3.tflite'
IMAGES_FOLDER = 'dataset'
OUTPUT_FOLDER = 'output'
MASK_THRESHOLD = 0.2
CONTOUR_THRESHOLD = 0.2
BLUR_THRESHOLD = 0.1
BLUR_KERNEL = (55, 55)



def resize_image(image):
    image_height, image_width = image.shape[:2]
    
    if image_height < image_width:
        new_width = DESIRED_WIDTH
        new_height = int(image_height / (image_width / DESIRED_WIDTH))
    else:
        new_height = DESIRED_HEIGHT
        new_width = int(image_width / (image_height / DESIRED_HEIGHT))
    
    return cv2.resize(image, (new_width, new_height))



def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Image saved: {output_path}")



def extract_contours(category_mask, confidence_threshold=CONTOUR_THRESHOLD):
    mask_data = category_mask.numpy_view().squeeze(-1)
    binary_mask = (mask_data > confidence_threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_height, mask_width = binary_mask.shape
    contours_image = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
    cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 2)
    
    return contours_image



def create_composite_image(original, mask, blur, contours):
    original_resized = resize_image(original)
    mask_resized = resize_image(mask)
    contours_resized = resize_image(contours)
    blur_resized = resize_image(blur)
    
    max_height = max(
        original_resized.shape[0],
        mask_resized.shape[0],
        contours_resized.shape[0],
        blur_resized.shape[0]
    )
    
    original_width = original_resized.shape[1]
    mask_width = mask_resized.shape[1]
    contours_width = contours_resized.shape[1]
    blur_width = blur_resized.shape[1]
    
    original_resized = cv2.resize(original_resized, (original_width, max_height))
    mask_resized = cv2.resize(mask_resized, (mask_width, max_height))
    contours_resized = cv2.resize(contours_resized, (contours_width, max_height))
    blur_resized = cv2.resize(blur_resized, (blur_width, max_height))
    
    def to_bgr(img):
        if len(img.shape) == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    
    original_resized = to_bgr(original_resized)
    mask_resized = to_bgr(mask_resized)
    contours_resized = to_bgr(contours_resized)
    blur_resized = to_bgr(blur_resized)
    
    return np.hstack([original_resized, mask_resized, contours_resized, blur_resized])



def create_segmenter(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    return vision.ImageSegmenter.create_from_options(options)


def segment_with_mask(segmenter, image_path, confidence_threshold=MASK_THRESHOLD):
    image = mp.Image.create_from_file(image_path)
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    
    image_data = image.numpy_view()
    image_height, image_width = image_data.shape[:2]
    num_channels = image_data.shape[2] if len(image_data.shape) > 2 else 1
    
    if num_channels == 4:
        mask_color = MASK_COLOR + (255,)
        bg_color = BG_COLOR + (255,)
    else:
        mask_color = MASK_COLOR[:num_channels]
        bg_color = BG_COLOR[:num_channels]
    
    object_image = np.full((image_height, image_width, num_channels), mask_color, dtype=np.uint8)
    background_image = np.full((image_height, image_width, num_channels), bg_color, dtype=np.uint8)
    
    mask_condition = category_mask.numpy_view().squeeze(-1) > confidence_threshold
    output_image = np.where(mask_condition[..., None], object_image, background_image)
    
    return output_image, segmentation_result


def segment_with_blur(segmenter, image_path, confidence_threshold=BLUR_THRESHOLD, blur_kernel=BLUR_KERNEL):
    image = mp.Image.create_from_file(image_path)
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    
    image_data = image.numpy_view()[:, :, :3]
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    blurred_image = cv2.GaussianBlur(image_data, blur_kernel, 0)
    
    mask_condition = category_mask.numpy_view().squeeze(-1) > confidence_threshold
    output_image = np.where(mask_condition[..., None], image_data, blurred_image)
    
    return output_image



def main(parent_folder=IMAGES_FOLDER):
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        print(f"Error: Folder '{parent_folder}' does not exist.")
        return
    
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)
    
    subfolders = [d for d in parent_path.iterdir() if d.is_dir()]
    if not subfolders:
        print(f"No subfolders found in '{parent_folder}'.")
        return
    
    print("Initializing model...")
    with create_segmenter(MODEL_PATH) as segmenter:
        for subfolder in subfolders:
            print(f"\nProcessing: {subfolder.name}")
            
            images = list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.jpeg')) + list(subfolder.glob('*.png'))
            if not images:
                print(f"No images found")
                continue
            
            print(f"Found {len(images)} images")
            subfolder_output = output_path / subfolder.name
            subfolder_output.mkdir(exist_ok=True)
            
            for image_file in images:
                original_image = cv2.imread(str(image_file))
                if original_image is None:
                    print(f"Failed to read: {image_file.name}")
                    continue
                
                print(f"Processing: {image_file.name}")
                
                mask_output, segmentation_result = segment_with_mask(segmenter, str(image_file))
                contours = extract_contours(segmentation_result.category_mask)
                blur_output = segment_with_blur(segmenter, str(image_file))
                composite = create_composite_image(original_image, mask_output, blur_output, contours)
                
                output_file = subfolder_output / f'{image_file.stem}.png'
                save_image(composite, str(output_file))
    
    print(f"\nDone! Results in '{output_path}'.")


if __name__ == '__main__':
    main()
