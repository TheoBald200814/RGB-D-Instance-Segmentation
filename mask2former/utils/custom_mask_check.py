import os
import numpy as np
import cv2
from PIL import Image # Using PIL to get image size reliably
import random
import argparse
import sys

# --- Configuration ---
# Define the channels in the generated 3-channel mask
MASK_INSTANCE_CHANNEL = 1 # Channel index for instance IDs
MASK_SEMANTIC_CHANNEL = 2 # Channel index for semantic IDs
MASK_BACKGROUND_ID = 0    # Pixel value for background

# Display settings
DISPLAY_HEIGHT = 400 # Target height for displayed images (aspect ratio will be maintained)
OVERLAY_ALPHA = 0.5  # Transparency for the mask overlay (0.0 to 1.0)

# --- Helper Functions ---

def create_color_map(id):
    """
    Generates a deterministic color for a given ID.
    ID 0 is black (background).
    """
    if id == MASK_BACKGROUND_ID:
        return (0, 0, 0) # Black for background

    # Generate a color based on the ID
    # Using a simple hash-like function to get distinct colors
    # These colors might not be perceptually uniform, but should be distinct for different IDs
    np.random.seed(id) # Use ID as seed for reproducibility
    color = np.random.randint(0, 256, size=3).tolist()
    # Ensure color is not too close to black or white if possible, or add some variation
    # A simple approach:
    # color = [(id * 123 + 50) % 200 + 55, (id * 456 + 100) % 200 + 55, (id * 789 + 150) % 200 + 55]
    # Let's stick to the simple random for now, seeded by ID.
    return tuple(color) # OpenCV uses BGR, but for visualization RGB is fine, cv2.cvtColor handles it if needed

def apply_color_map_to_mask(mask, color_map_func):
    """
    Applies a color map function to a single-channel mask.
    Returns a 3-channel color image.
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    unique_ids = np.unique(mask)

    for id in unique_ids:
        color = color_map_func(id)
        # Find pixels with this ID and set their color
        color_mask[mask == id] = color

    return color_mask

def resize_to_height(image, target_height):
    """Resizes an image to a target height while maintaining aspect ratio."""
    if image.shape[0] == target_height:
        return image
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)
    # Ensure width is at least 1 pixel
    target_width = max(1, target_width)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def pad_image(image, target_width, target_height, pad_value=0):
    """Pads an image to a target width and height."""
    h, w = image.shape[:2]
    if h > target_height or w > target_width:
        raise ValueError("Image is larger than target dimensions after resizing.")

    pad_h = target_height - h
    pad_w = target_width - w

    # Pad on bottom and right
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return padded_image

def visualize_masks(image_folder, mask_folder, num_images_to_visualize=5):
    """
    Visualizes original images and their corresponding generated masks.

    Args:
        image_folder (str): Path to the folder containing original images.
        mask_folder (str): Path to the folder containing generated 3-channel masks.
        num_images_to_visualize (int): Number of images to visualize.
    """
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    if not image_files:
        print(f"No image files found in {image_folder}")
        return

    # Assuming mask files are PNG and have the same base name as image files
    mask_files = [os.path.splitext(f)[0] + '.png' for f in image_files]

    # Filter out images that don't have a corresponding mask file
    available_images = []
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            available_images.append((img_file, mask_file))
        else:
            # print(f"Skipping {img_file}: corresponding mask {mask_file} not found.")
            pass # Silently skip if mask is missing

    if not available_images:
        print(f"No matching image and mask files found between {image_folder} and {mask_folder}")
        return

    # Select random images to visualize
    num_to_show = min(num_images_to_visualize, len(available_images))
    selected_images = random.sample(available_images, num_to_show)

    print(f"Visualizing {num_to_show} random images...")

    for img_file, mask_file in selected_images:
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)

        print(f"\nProcessing {img_file}...")

        # Load original image (as color)
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error loading image {img_path}. Skipping.")
            continue

        # Load mask (unchanged to preserve channels and depth)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            print(f"Error loading mask {mask_path}. Skipping.")
            continue

        # Check mask format
        if mask_image.ndim != 3 or mask_image.shape[2] != 3:
            print(f"Mask {mask_path} is not a 3-channel image. Skipping.")
            continue
        if mask_image.shape[:2] != original_image.shape[:2]:
             print(f"Mask {mask_path} size {mask_image.shape[:2]} does not match image size {original_image.shape[:2]}. Skipping.")
             continue

        # Extract instance and semantic channels
        instance_mask = mask_image[:, :, MASK_INSTANCE_CHANNEL]
        semantic_mask = mask_image[:, :, MASK_SEMANTIC_CHANNEL]

        # --- Create Visualizations ---

        # 1. Original Image (already loaded)

        # 2. Colored Semantic Mask
        colored_semantic_mask = apply_color_map_to_mask(semantic_mask, create_color_map)

        # 3. Colored Instance Mask
        colored_instance_mask = apply_color_map_to_mask(instance_mask, create_color_map)

        # 4. Semantic Overlay
        # Convert images to float for blending
        original_image_float = original_image.astype(np.float32) / 255.0
        colored_semantic_mask_float = colored_semantic_mask.astype(np.float32) / 255.0
        semantic_overlay = cv2.addWeighted(original_image_float, 1.0 - OVERLAY_ALPHA, colored_semantic_mask_float, OVERLAY_ALPHA, 0)
        semantic_overlay = (semantic_overlay * 255.0).astype(np.uint8)

        # 5. Instance Overlay
        original_image_float = original_image.astype(np.float32) / 255.0 # Reload/re-convert if needed
        colored_instance_mask_float = colored_instance_mask.astype(np.float32) / 255.0
        instance_overlay = cv2.addWeighted(original_image_float, 1.0 - OVERLAY_ALPHA, colored_instance_mask_float, OVERLAY_ALPHA, 0)
        instance_overlay = (instance_overlay * 255.0).astype(np.uint8)


        # --- Prepare Images for Display ---
        images_to_display = [
            original_image,
            colored_semantic_mask,
            colored_instance_mask,
            semantic_overlay,
            instance_overlay
        ]
        labels = [
            "Original",
            "Semantic Mask",
            "Instance Mask",
            "Semantic Overlay",
            "Instance Overlay"
        ]

        # Resize all images to a common height and pad to a common width for concatenation
        resized_images = [resize_to_height(img, DISPLAY_HEIGHT) for img in images_to_display]
        max_width = max(img.shape[1] for img in resized_images)

        padded_images = [pad_image(img, max_width, DISPLAY_HEIGHT) for img in resized_images]

        # Add labels to images
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255) # White text

        labeled_images = []
        for img, label in zip(padded_images, labels):
            # Draw text with a black outline for better visibility
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x = 10 # Padding from left
            text_y = text_h + 10 # Padding from top

            # Draw black outline
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
            # Draw white text
            cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            labeled_images.append(img)


        # Concatenate images into a grid (e.g., 2 rows, 3 columns, last spot empty)
        # Row 1: Original, Colored Semantic, Colored Instance
        # Row 2: Semantic Overlay, Instance Overlay, Empty
        row1 = cv2.hconcat(labeled_images[:3])
        # Create an empty placeholder for the 6th spot
        empty_placeholder = np.full((DISPLAY_HEIGHT, max_width, 3), 50, dtype=np.uint8) # Grey placeholder
        row2 = cv2.hconcat([labeled_images[3], labeled_images[4], empty_placeholder])

        combined_image = cv2.vconcat([row1, row2])


        # --- Display ---
        window_name = f"Visualization: {img_file}"
        cv2.imshow(window_name, combined_image)

        # Wait for a key press. Press 'q' to quit, any other key for the next image.
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return # Exit the function

    # Destroy all windows after the loop finishes
    cv2.destroyAllWindows()


# --- Main Execution ---

if __name__ == "__main__":
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        print("Running visualization with command-line arguments.")
        parser = argparse.ArgumentParser(description="Visualize original images and converted 3-channel masks.")
        parser.add_argument("--image_folder", required=True,
                            help="Directory containing the original images.")
        parser.add_argument("--mask_folder", required=True,
                            help="Directory containing the generated 3-channel mask images.")
        parser.add_argument("--num_images", type=int, default=5,
                            help="Number of random images to visualize (default: 5).")

        args = parser.parse_args()
    else:
        print("Running visualization in IDE without command-line arguments. Using hardcoded defaults.")
        # If no arguments, manually set default values for IDE testing
        class IDEArgs:
            pass
        args = IDEArgs()

        # --- 在这里设置你的默认参数值 ---
        # !!! IMPORTANT: 请根据你的实际情况修改下面的路径 !!!
        args.image_folder = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/test/images' # 替换为你的原始图像文件夹路径
        args.mask_folder = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/test/processed_masks_48' # 替换为你保存转换后 mask 的文件夹路径
        args.num_images = 10 # 要可视化的图片数量

        # --- 默认参数设置结束 ---

    # Run the visualization
    try:
        visualize_masks(
            image_folder=args.image_folder,
            mask_folder=args.mask_folder,
            num_images_to_visualize=args.num_images
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the image folder and mask folder exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

