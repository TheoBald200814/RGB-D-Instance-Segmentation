import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import datetime # Added for COCO info

import numpy as np
from PIL import Image
import cv2 # Requires opencv-python
from pycocotools.coco import COCO # Requires pycocotools
from pycocotools import mask as coco_mask
from tqdm import tqdm
import argparse

# --- Configuration ---
# Define the target mask format properties
TARGET_MASK_CHANNELS = 3
TARGET_INSTANCE_CHANNEL = 1 # Channel index for instance IDs
TARGET_SEMANTIC_CHANNEL = 2 # Channel index for semantic IDs
TARGET_BACKGROUND_ID = 0    # Pixel value for background in both instance and semantic channels

# Define a mapping from source dataset category IDs/names to target semantic IDs
# Target semantic IDs MUST be >= 1 for actual classes. Source background should map to 0.
# Example:
# {
#   "source_dataset_name_1": {
#     "source_category_name_A": 1, # Map source name A to target semantic ID 1
#     "source_category_name_B": 2  # Map source name B to target semantic ID 2
#   },
#   "source_dataset_name_2": {
#     "source_category_id_X": 1, # Map source ID X to target semantic ID 1
#     "source_category_id_Y": 3  # Map source ID Y to target semantic ID 3
#   },
#   "background": 0 # Example mapping for a source background class
#   ...
# }
# Or a simpler flat mapping if source names/ids are unique across datasets:
# {
#   "person": 1,
#   "car": 2,
#   "bus": 3,
#   "road": 4, # Example semantic class
#   "sky": 5,  # Example semantic class
#   "background": 0 # Map any source background to target 0
#   ...
# }
# You NEED to create this mapping file based on your datasets and target categories.
CATEGORY_MAPPING_FILE = 'category_mapping.json'

# --- Helper Functions ---

def get_image_size(image_path):
    """Get image size (width, height) using Pillow."""
    try:
        with Image.open(image_path) as img:
            return img.size # Returns (width, height)
    except Exception as e:
        print(f"Error getting size for {image_path}: {e}")
        return None

def polygon_to_mask(segmentation, height, width):
    """Convert COCO polygon format to binary mask."""
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    # pycocotools expects list of polygons or RLEs
    rles = coco_mask.frPyObjects(segmentation, height, width)
    mask = coco_mask.decode(rles) # Decode RLEs into a binary mask (H, W)
    # If multiple polygons for one instance, decode might return (H, W, N), sum them
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8) # Ensure binary mask

def rle_to_mask(segmentation, height, width):
    """Convert COCO RLE format to binary mask."""
    if not segmentation:
         return np.zeros((height, width), dtype=np.uint8)
    # pycocotools expects list of RLEs, even if it's just one
    if isinstance(segmentation, dict):
        segmentation = [segmentation]
    mask = coco_mask.decode(segmentation) # Decode RLEs into a binary mask (H, W)
    # If multiple RLEs for one instance, decode might return (H, W, N), sum them
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8) # Ensure binary mask

def mask_to_polygon(binary_mask):
    """
    Convert a binary mask to COCO polygon format.
    Handles multiple disconnected regions and holes.

    Args:
        binary_mask (np.ndarray): A 2D numpy array (H, W) of dtype uint8,
                                  where pixels belonging to the object are > 0.

    Returns:
        list: A list of flattened polygon coordinates [x1, y1, x2, y2, ...].
              Returns an empty list if no contours are found or mask is empty.
    """
    # Ensure mask is binary (0 or 255) for findContours
    mask = binary_mask.astype(np.uint8) * 255

    # Find contours using RETR_TREE to get hierarchy (for holes)
    # Use CHAIN_APPROX_SIMPLE to compress horizontal, vertical, and diagonal segments
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    segmentation = []
    # hierarchy is (num_contours, 1, 4) array: [Next, Previous, First_Child, Parent]
    # We only care about outer contours (Parent == -1) and their children (holes)
    # Iterate through contours and their hierarchy
    for i, contour in enumerate(contours):
        # Check if it's an outer contour (no parent)
        if hierarchy[0][i][3] == -1:
            # This is an outer boundary. Add its points.
            # Reshape from (N, 1, 2) to (N, 2) and flatten
            polygon = contour.reshape(-1).tolist()

            # Check for holes (children)
            child_idx = hierarchy[0][i][2] # Index of the first child
            while child_idx != -1:
                # This is a hole. Add its points.
                hole_contour = contours[child_idx]
                hole_polygon = hole_contour.reshape(-1).tolist()
                polygon.extend(hole_polygon) # Add hole points to the same segmentation list

                # Move to the next sibling hole
                child_idx = hierarchy[0][child_idx][0] # Index of the next contour at the same level

            # COCO requires polygons with at least 6 points (3 vertices)
            if len(polygon) >= 6:
                 segmentation.append(polygon)

    # Filter out empty segmentations that might result from small contours
    segmentation = [poly for poly in segmentation if poly]

    return segmentation


# --- Converter Class ---

class AnnotationConverter:
    def __init__(self, input_format, input_dir, output_dir, category_mapping_file, image_subdir="images"):
        self.input_format = input_format.lower()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.image_dir = os.path.join(input_dir, image_subdir) # Assuming images are in a subdir
        self.category_mapping = self._load_category_mapping(category_mapping_file)

        os.makedirs(self.output_dir, exist_ok=True)

        # Map format strings to parsing methods
        self.parser_map = {
            'coco': self._parse_coco,
            'separate_masks': self._parse_separate_masks,
            # Add other formats here
        }

        if self.input_format not in self.parser_map:
            raise ValueError(f"Unsupported input format: {self.input_format}. Supported formats: {list(self.parser_map.keys())}")

    def _load_category_mapping(self, mapping_file):
        """Loads the category mapping from a JSON file."""
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Category mapping file not found: {mapping_file}")
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)

        # Validate mapping: ensure target semantic IDs are >= 0
        for source_cat, target_id in mapping.items():
            if not isinstance(target_id, int) or target_id < 0:
                 raise ValueError(f"Invalid target semantic ID '{target_id}' for source category '{source_cat}' in mapping file. Target IDs must be non-negative integers.")
            if target_id == 0 and source_cat != "background" and source_cat != str(TARGET_BACKGROUND_ID):
                 print(f"Warning: Source category '{source_cat}' is mapped to target ID 0. Ensure this is intended for background/ignored regions.")

        print(f"Loaded category mapping from {mapping_file}")
        # print(f"Mapping: {mapping}") # Optional: print mapping for verification
        return mapping

    def _get_target_semantic_id(self, source_category_info):
        """Looks up the target semantic ID using the category mapping."""
        # source_category_info can be a name (str) or an ID (int)
        # Try looking up as string first, then as string representation of int
        target_id = self.category_mapping.get(str(source_category_info))
        if target_id is None:
             # Also try looking up if the source info is an int and mapped directly
             if isinstance(source_category_info, int):
                  target_id = self.category_mapping.get(source_category_info) # Check if int key exists
             if target_id is None:
                  # Check if source_category_info is a string that represents an int
                  try:
                       int_key = int(source_category_info)
                       target_id = self.category_mapping.get(int_key)
                  except (ValueError, TypeError):
                       pass # Not an int string

        if target_id is None:
            # Handle unmapped categories - return a special value or raise error
            # For now, let's return None and the main loop will handle skipping/warning
            return None
        return target_id

    def _parse_coco(self, annotation_file="annotations.json"):
        """Parses COCO format and yields annotation data per image."""
        annotation_path = os.path.join(self.input_dir, annotation_file)
        if not os.path.exists(annotation_path):
             raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        print(f"Parsing COCO format from {annotation_path}")
        coco = COCO(annotation_path)
        img_ids = coco.getImgIds()

        # Create a mapping from COCO category ID to name for lookup
        coco_cats = coco.loadCats(coco.getCatIds())
        coco_id_to_name = {cat['id']: cat['name'] for cat in coco_cats}

        for img_id in tqdm(img_ids, desc="Parsing COCO images"):
            img_info = coco.loadImgs(img_id)[0]
            image_filename = img_info['file_name']
            image_path = os.path.join(self.image_dir, image_filename) # Use configured image_dir

            # Get image size
            img_size = get_image_size(image_path)
            if img_size is None:
                 print(f"Skipping image {image_filename} due to size error.")
                 continue
            img_width, img_height = img_size

            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)

            # Prepare annotations for this image
            image_annotations = []
            for ann in annotations:
                # Get binary mask from segmentation (polygon or RLE)
                if 'segmentation' not in ann or not ann['segmentation']:
                     # print(f"Warning: Annotation {ann['id']} in image {image_filename} has no segmentation. Skipping.")
                     continue # Skip annotations without segmentation

                if isinstance(ann['segmentation'], list): # Polygon format
                    binary_mask = polygon_to_mask(ann['segmentation'], img_height, img_width)
                elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']: # RLE format
                    binary_mask = rle_to_mask(ann['segmentation'], img_height, img_width)
                else:
                    print(f"Warning: Unknown segmentation format for annotation {ann['id']} in image {image_filename}. Skipping.")
                    continue

                # Get source category info (use name if available, otherwise ID)
                source_cat_id = ann['category_id']
                source_cat_info = coco_id_to_name.get(source_cat_id, source_cat_id) # Prefer name, fallback to ID

                image_annotations.append({
                    "mask": binary_mask,
                    "source_category_info": source_cat_info,
                    "iscrowd": ann.get('iscrowd', 0) # Default iscrowd to 0 if not present
                })

            yield image_filename, img_width, img_height, image_annotations


    def _parse_separate_masks(self, instance_mask_subdir="instance_masks", semantic_mask_subdir="semantic_masks", mask_ext=".png"):
        """
        Parses format with separate instance and semantic mask files per image.
        Assumes mask files have the same base name as the image file.
        e.g., image.jpg -> instance_masks/image.png, semantic_masks/image.png
        Instance IDs might be local to semantic classes.
        """
        instance_masks_dir = os.path.join(self.input_dir, instance_mask_subdir)
        semantic_masks_dir = os.path.join(self.input_dir, semantic_mask_subdir)

        if not os.path.exists(instance_masks_dir):
             raise FileNotFoundError(f"Instance masks directory not found: {instance_masks_dir}")
        if not os.path.exists(semantic_masks_dir):
             raise FileNotFoundError(f"Semantic masks directory not found: {semantic_masks_dir}")

        image_files = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]

        for image_filename in tqdm(image_files, desc="Parsing separate mask files"):
            image_path = os.path.join(self.image_dir, image_filename)
            base_name = os.path.splitext(image_filename)[0]

            instance_mask_path = os.path.join(instance_masks_dir, base_name + mask_ext)
            semantic_mask_path = os.path.join(semantic_masks_dir, base_name + mask_ext)

            if not os.path.exists(instance_mask_path):
                # print(f"Warning: Instance mask not found for {image_filename} at {instance_mask_path}. Skipping.")
                continue
            if not os.path.exists(semantic_mask_path):
                # print(f"Warning: Semantic mask not found for {image_filename} at {semantic_mask_path}. Skipping.")
                continue

            # Get image size
            img_size = get_image_size(image_path)
            if img_size is None:
                 print(f"Skipping image {image_filename} due to size error.")
                 continue
            img_width, img_height = img_size

            # Load mask images
            try:
                # Use 'I' for 32-bit signed integer, 'I;16' for 16-bit unsigned integer PNG
                # 'L' for 8-bit grayscale. Instance/semantic IDs can be large, prefer 16-bit or 32-bit.
                # Let's try 16-bit first, fallback to 8-bit if needed.
                try:
                    instance_mask_img = Image.open(instance_mask_path).convert('I;16')
                    semantic_mask_img = Image.open(semantic_mask_path).convert('I;16')
                except IOError: # Fallback for non-16bit PNGs or other formats
                    instance_mask_img = Image.open(instance_mask_path).convert('L')
                    semantic_mask_img = Image.open(semantic_mask_path).convert('L')


                instance_mask_array = np.array(instance_mask_img)
                semantic_mask_array = np.array(semantic_mask_img)

                # Ensure mask sizes match image size
                if instance_mask_array.shape[:2] != (img_height, img_width) or \
                   semantic_mask_array.shape[:2] != (img_height, img_width):
                    print(f"Warning: Mask sizes for {image_filename} do not match image size ({instance_mask_array.shape[:2]} vs {(img_height, img_width)}). Skipping.")
                    continue

            except Exception as e:
                print(f"Error loading mask files for {image_filename}: {e}. Skipping.")
                continue

            image_annotations = []

            # --- Process Instances ---
            # Find unique (semantic_id, local_instance_id) pairs that are not background
            # Stack the two masks to get pairs for each pixel
            combined_mask = np.stack([semantic_mask_array, instance_mask_array], axis=-1) # Shape (H, W, 2)
            # Reshape to (H*W, 2) and find unique rows
            unique_pairs = np.unique(combined_mask.reshape(-1, 2), axis=0)

            # Filter out the background pair (0, 0)
            instance_pairs = unique_pairs[(unique_pairs[:, 0] != TARGET_BACKGROUND_ID) | (unique_pairs[:, 1] != TARGET_BACKGROUND_ID)]
            instance_pairs = instance_pairs[instance_pairs[:, 1] != TARGET_BACKGROUND_ID] # Also explicitly remove pairs where instance_id is 0

            # Create a mask of all instance pixels (where instance_id > 0)
            all_instances_mask = (instance_mask_array > TARGET_BACKGROUND_ID)

            for sem_id, local_inst_id in instance_pairs:
                # Create binary mask for this specific (semantic_id, local_instance_id) combination
                binary_mask = ((semantic_mask_array == sem_id) & (instance_mask_array == local_inst_id)).astype(np.uint8)

                if np.sum(binary_mask) == 0:
                     # This should ideally not happen if unique_pairs are from the mask, but as a safeguard
                     print(f"Warning: Generated empty mask for pair ({sem_id}, {local_inst_id}) in {image_filename}. Skipping.")
                     continue

                # Map source semantic ID to target semantic ID
                target_semantic_id = self._get_target_semantic_id(sem_id)

                if target_semantic_id is None:
                     print(f"Warning: Source semantic ID {sem_id} for instance ({sem_id}, {local_inst_id}) in {image_filename} not found in mapping. Skipping instance.")
                     continue
                # Note: An instance should ideally not map to background semantic ID 0.
                # If it does, it will be treated as an instance with semantic ID 0 in the output.
                # This might be valid if 0 is used for 'unlabeled' or similar.

                image_annotations.append({
                    "mask": binary_mask,
                    "source_category_info": sem_id, # Store source semantic ID for mapping lookup
                    "iscrowd": 0, # These are instances
                    # We don't store the local_inst_id here, a new global ID will be assigned in convert()
                })

            # --- Process Stuff (Semantic Regions not covered by Instances) ---
            # Iterate through unique semantic IDs in the semantic mask (excluding background)
            unique_semantic_ids = np.unique(semantic_mask_array)

            for sem_id in unique_semantic_ids:
                 if sem_id == TARGET_BACKGROUND_ID: # Skip background semantic ID
                      continue

                 # Map source semantic ID to target semantic ID
                 target_semantic_id = self._get_target_semantic_id(sem_id)

                 if target_semantic_id is None:
                      # print(f"Warning: Source semantic ID {sem_id} in {image_filename} not found in mapping for stuff processing. Skipping.")
                      continue # Skip unmapped semantic IDs

                 # Create a mask for the current semantic region
                 semantic_region_mask = (semantic_mask_array == sem_id).astype(np.uint8)

                 # Identify pixels in this semantic region that are NOT part of any instance (where instance_id is 0)
                 # This assumes instance masks are strictly on top of stuff or background.
                 stuff_mask = ((semantic_region_mask > 0) & (instance_mask_array == TARGET_BACKGROUND_ID)).astype(np.uint8)

                 if np.sum(stuff_mask) == 0:
                      # print(f"Info: No stuff pixels found for semantic ID {sem_id} in {image_filename} after removing instance areas.")
                      continue # No stuff region for this semantic ID

                 # Find connected components in the stuff mask
                 num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(stuff_mask, 8, cv2.CV_32S)

                 # Iterate through each connected component (excluding background label 0)
                 for i in range(1, num_labels):
                      component_mask = (labels_img == i).astype(np.uint8)

                      if np.sum(component_mask) == 0:
                           continue # Should not happen for labels 1 to num_labels-1, but safe check

                      # Add this component as a stuff annotation
                      image_annotations.append({
                           "mask": component_mask,
                           "source_category_info": sem_id, # Store source semantic ID
                           "iscrowd": 1, # Treat as stuff
                      })


            yield image_filename, img_width, img_height, image_annotations


    # Add more parsing methods here for other formats (e.g., _parse_custom_json)
    # def _parse_custom_json(self, annotation_file="custom_annotations.json"):
    #     """Parses a custom JSON format."""
    #     annotation_path = os.path.join(self.input_dir, annotation_file)
    #     if not os.path.exists(annotation_path):
    #          raise FileNotFoundError(f"Custom annotation file not found: {annotation_path}")
    #
    #     print(f"Parsing custom JSON format from {annotation_path}")
    #     with open(annotation_path, 'r') as f:
    #         custom_data = json.load(f)
    #
    #     # --- CUSTOM PARSING LOGIC ---
    #     # You need to write code here to iterate through images and their annotations
    #     # and yield data in the format: image_filename, img_width, img_height, image_annotations
    #     # where image_annotations is a list of dicts like:
    #     # { "mask": binary_mask_array, "source_category_info": source_cat_id_or_name, "iscrowd": 0 or 1 }
    #
    #     # Example structure (adapt to your JSON):
    #     # for img_data in tqdm(custom_data["images"], desc="Parsing custom JSON images"):
    #     #     image_filename = img_data["file_name"]
    #     #     image_path = os.path.join(self.image_dir, image_filename)
    #     #     img_size = get_image_size(image_path)
    #     #     if img_size is None: continue
    #     #     img_width, img_height = img_size
    #     #
    #     #     image_annotations = []
    #     #     for ann_data in img_data.get("annotations", []):
    #     #         # Assuming segmentation is polygon or RLE, or maybe a mask file path?
    #     #         # Need to convert to binary_mask_array (H, W, uint8)
    #     #         # Need to get source_category_info (name or ID)
    #     #         # Need to get iscrowd (default to 0 if not present)
    #     #
    #     #         # Example: Assuming segmentation is a list of polygons
    #     #         segmentation = ann_data.get("segmentation", [])
    #     #         if isinstance(segmentation, list):
    #     #              binary_mask = polygon_to_mask(segmentation, img_height, img_width)
    #     #         # Add logic for RLE or mask file paths if needed
    #     #         else:
    #     #              print(f"Warning: Unknown segmentation format for annotation in {image_filename}. Skipping.")
    #     #              continue
    #     #
    #     #         source_cat_info = ann_data.get("category_id") # Or ann_data.get("category_name")
    #     #         iscrowd = ann_data.get("iscrowd", 0)
    #     #
    #     #         if binary_mask is not None and np.sum(binary_mask) > 0: # Ensure mask is valid
    #     #              image_annotations.append({
    #     #                  "mask": binary_mask,
    #     #                  "source_category_info": source_cat_info,
    #     #                  "iscrowd": iscrowd
    #     #              })
    #     #
    #     #     yield image_filename, img_width, img_height, image_annotations
    #     # --- END CUSTOM PARSING LOGIC ---


    def convert(self):
        """Performs the conversion for all images in the input directory."""
        parse_method = self.parser_map[self.input_format]
        image_data_generator = parse_method() # Get the generator

        unmapped_categories = set()
        processed_images_count = 0

        for image_filename, img_width, img_height, image_annotations in image_data_generator:
            processed_images_count += 1
            output_mask_filename = os.path.splitext(image_filename)[0] + '.png' # Save as PNG
            output_mask_path = os.path.join(self.output_dir, output_mask_filename)

            # Create the target 3-channel mask image initialized with background (0)
            # Use uint16 for instance IDs as they can be large. Semantic IDs might also be large.
            # OpenCV imwrite can handle uint16 for PNG.
            target_mask = np.full((img_height, img_width, TARGET_MASK_CHANNELS), TARGET_BACKGROUND_ID, dtype=np.uint16)

            current_instance_id = 1 # Global instance IDs start from 1

            # Process annotations for this image
            # It's good practice to process stuff first, then instances,
            # so instances overwrite stuff in case of overlap (common convention).
            # Sort annotations by iscrowd (1 comes before 0)
            image_annotations.sort(key=lambda x: x['iscrowd'], reverse=True)


            for ann in image_annotations:
                binary_mask = ann["mask"]
                source_cat_info = ann["source_category_info"]
                iscrowd = ann["iscrowd"]

                # Get target semantic ID
                target_semantic_id = self._get_target_semantic_id(source_cat_info)

                if target_semantic_id is None:
                    unmapped_categories.add(source_cat_info)
                    # print(f"Warning: Source category '{source_cat_info}' for image {image_filename} not in mapping. Skipping annotation.")
                    continue # Skip annotations with unmapped categories

                # Find pixels belonging to this annotation's mask
                mask_pixels_y, mask_pixels_x = np.where(binary_mask > 0)

                if mask_pixels_y.size == 0:
                     # print(f"Warning: Binary mask for source category '{source_cat_info}' in image {image_filename} is empty. Skipping.")
                     continue # Skip empty masks

                # Apply semantic ID to channel 2 for ALL annotations (instance and stuff)
                target_mask[mask_pixels_y, mask_pixels_x, TARGET_SEMANTIC_CHANNEL] = target_semantic_id

                # Apply instance ID to channel 1 ONLY if it's an instance (iscrowd == 0)
                if iscrowd == 0:
                    # Assign the current global instance ID
                    target_mask[mask_pixels_y, mask_pixels_x, TARGET_INSTANCE_CHANNEL] = current_instance_id
                    # Increment the global instance ID counter for the next instance
                    current_instance_id += 1
                else:
                    # iscrowd == 1 (stuff/semantic region)
                    # Instance channel remains 0 for these pixels (or is overwritten by instances later)
                    pass # No need to set channel 1, it's already 0 or will be set by an instance


            # Channel 0 remains 0 as initialized

            # Save the resulting mask image
            try:
                # Use OpenCV to save PNG. It handles uint16 for PNG.
                # The channels are [0, instance, semantic]. OpenCV saves BGR.
                # So channel 0 -> B, channel 1 -> G, channel 2 -> R.
                # This is fine as long as the loader knows which channel is which.
                # We can save it directly.
                cv2.imwrite(output_mask_path, target_mask)
                # print(f"Saved mask for {image_filename} to {output_mask_path}")
            except Exception as e:
                print(f"Error saving mask for {image_filename} to {output_mask_path}: {e}")


        # Report unmapped categories
        if unmapped_categories:
            print("\n--- WARNING: Unmapped Source Categories ---")
            print("The following source categories were found but not present in your category_mapping.json:")
            for cat in unmapped_categories:
                print(f"- {cat}")
            print("Annotations with these categories were skipped during mask generation.")
            print("-------------------------------------------\n")

        print(f"\nConversion complete. Processed {processed_images_count} images.")
        print(f"Converted masks saved to {self.output_dir}")

    # --- NEW MEMBER FUNCTION: Count Instances ---
    def count_instances_in_masks(self, mask_dir=None):
        """
        Counts the number of instances for each category present in the converted
        3-channel mask files.

        Args:
            mask_dir (str, optional): Directory containing the 3-channel mask files.
                                      If None, uses the output_dir specified during initialization.
                                      Defaults to None.

        Returns:
            dict: A dictionary where keys are target semantic IDs (integers) and
                  values are the total count of instances for that category across
                  all processed masks.
        """
        target_mask_dir = mask_dir if mask_dir is not None else self.output_dir

        if not os.path.isdir(target_mask_dir):
            print(f"Error: Mask directory not found or is not a directory: {target_mask_dir}")
            return {}

        print(f"\nCounting instances in masks from: {target_mask_dir}")

        instance_counts = defaultdict(int)
        mask_files = [f for f in os.listdir(target_mask_dir) if f.endswith('.png')] # Assuming PNG output

        if not mask_files:
            print(f"No PNG mask files found in {target_mask_dir}.")
            return {}

        # Create an inverse mapping from target semantic ID to source category info
        # This is for reporting purposes, to show category names instead of just IDs
        target_id_to_source_cat = {v: k for k, v in self.category_mapping.items()}


        for mask_filename in tqdm(mask_files, desc="Counting instances"):
            mask_path = os.path.join(target_mask_dir, mask_filename)

            try:
                # Use OpenCV to load the 3-channel mask, preserving uint16 depth
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if mask is None:
                    print(f"Warning: Could not load mask file {mask_filename}. Skipping.")
                    continue

                # Ensure it's a 3-channel mask as expected
                if mask.ndim != 3 or mask.shape[2] != TARGET_MASK_CHANNELS:
                    print(f"Warning: Mask file {mask_filename} is not {TARGET_MASK_CHANNELS} channels ({mask.ndim} dims or {mask.shape[2]} channels). Skipping.")
                    continue

                # Extract the instance and semantic channels
                instance_channel = mask[:, :, TARGET_INSTANCE_CHANNEL]
                semantic_channel = mask[:, :, TARGET_SEMANTIC_CHANNEL]

                # Find all unique non-zero instance IDs in this mask
                unique_instance_ids = np.unique(instance_channel)
                # Filter out the background instance ID (0)
                instance_ids_in_mask = unique_instance_ids[unique_instance_ids != TARGET_BACKGROUND_ID]

                # For each unique instance ID, find its corresponding semantic ID and increment count
                for inst_id in instance_ids_in_mask:
                    # Find the location of this instance ID
                    # We only need one pixel location to get the semantic ID
                    locations = np.where(instance_channel == inst_id)

                    if locations[0].size > 0: # Ensure the instance ID actually exists in the mask
                        # Get the semantic ID at the first pixel location of this instance
                        sem_id_at_instance_loc = semantic_channel[locations[0][0], locations[1][0]]

                        # Increment the count for this target semantic ID
                        instance_counts[sem_id_at_instance_loc] += 1
                    else:
                         # This case should ideally not happen if unique_instance_ids comes from the mask
                         print(f"Warning: Instance ID {inst_id} found in unique list but not in mask pixels for {mask_filename}. Skipping count for this instance.")


            except Exception as e:
                print(f"Error processing mask file {mask_filename} for instance counting: {e}. Skipping.")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed error


        # --- Report the counts ---
        print("\n--- Instance Counts per Category ---")
        if not instance_counts:
            print("No instances found in the processed masks.")
        else:
            # Sort counts by target semantic ID for consistent reporting
            sorted_counts = sorted(instance_counts.items())
            for target_id, count in sorted_counts:
                # Get the source category name for reporting
                category_name = target_id_to_source_cat.get(target_id, f"Target_ID_{target_id}")
                print(f"Target ID {target_id} ({category_name}): {count} instances")
        print("------------------------------------\n")

        return dict(instance_counts) # Return as a regular dictionary

    # --- NEW MEMBER FUNCTION: Convert to COCO JSON ---
    def convert_to_coco_json(self, output_json_path, mask_dir=None):
        """
        Converts the 3-channel mask files into a COCO format JSON annotation file.
        Segmentation is represented as polygons.

        Args:
            output_json_path (str): The path where the output COCO JSON file will be saved.
            mask_dir (str, optional): Directory containing the 3-channel mask files.
                                      If None, uses the output_dir specified during initialization.
                                      Defaults to None.
        """
        target_mask_dir = mask_dir if mask_dir is not None else self.output_dir

        if not os.path.isdir(target_mask_dir):
            print(f"Error: Mask directory not found or is not a directory: {target_mask_dir}")
            return

        print(f"\nConverting masks from {target_mask_dir} to COCO JSON: {output_json_path}")

        coco_output = {
            "info": {
                "description": "Converted from 3-channel masks",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "date_created": datetime.datetime.now().strftime("%Y/%m/%d"),
            },
            "licenses": [], # Add license info if available
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Prepare categories list for COCO JSON
        # COCO categories typically don't include background (ID 0)
        # Create an inverse mapping from target_id to source_info for category names
        target_id_to_source_info = {v: k for k, v in self.category_mapping.items()}

        # Sort categories by target ID for consistent output
        sorted_target_ids = sorted([v for v in self.category_mapping.values() if v != TARGET_BACKGROUND_ID])

        for target_id in sorted_target_ids:
             source_info = target_id_to_source_info.get(target_id, f"Target_ID_{target_id}")
             # Use the source info as the category name. If it was an int, it will be a string.
             category_name = str(source_info)
             coco_output["categories"].append({
                 "id": target_id,
                 "name": category_name,
                 "supercategory": "none" # Or derive from mapping if available
             })

        image_id_counter = 0
        annotation_id_counter = 0

        mask_files = [f for f in os.listdir(target_mask_dir) if f.endswith('.png')] # Assuming PNG output

        if not mask_files:
            print(f"No PNG mask files found in {target_mask_dir}. No COCO JSON will be generated.")
            return

        for mask_filename in tqdm(mask_files, desc="Generating COCO annotations"):
            mask_path = os.path.join(target_mask_dir, mask_filename)
            # Assume original image filename is the same base name
            image_filename = os.path.splitext(mask_filename)[0] + os.path.splitext(os.listdir(self.image_dir)[0])[1] # Try to guess original extension

            # Get image size from the original image file
            image_path = os.path.join(self.image_dir, image_filename)
            img_size = get_image_size(image_path)
            if img_size is None:
                 print(f"Skipping COCO annotation for {image_filename} due to size error.")
                 continue
            img_width, img_height = img_size

            # Add image info
            image_id_counter += 1
            current_image_id = image_id_counter
            coco_output["images"].append({
                "id": current_image_id,
                "width": img_width,
                "height": img_height,
                "file_name": image_filename
            })

            try:
                # Use OpenCV to load the 3-channel mask, preserving uint16 depth
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if mask is None:
                    print(f"Warning: Could not load mask file {mask_filename} for COCO conversion. Skipping.")
                    continue

                # Ensure it's a 3-channel mask as expected
                if mask.ndim != 3 or mask.shape[2] != TARGET_MASK_CHANNELS:
                    print(f"Warning: Mask file {mask_filename} is not {TARGET_MASK_CHANNELS} channels ({mask.ndim} dims or {mask.shape[2]} channels) for COCO conversion. Skipping.")
                    continue

                # Extract the instance and semantic channels
                instance_channel = mask[:, :, TARGET_INSTANCE_CHANNEL]
                semantic_channel = mask[:, :, TARGET_SEMANTIC_CHANNEL]

                # --- Process Instances (iscrowd=0) ---
                unique_instance_ids = np.unique(instance_channel)
                instance_ids_in_mask = unique_instance_ids[unique_instance_ids != TARGET_BACKGROUND_ID]

                for inst_id in instance_ids_in_mask:
                    instance_mask = (instance_channel == inst_id).astype(np.uint8)

                    # Find the corresponding semantic ID for this instance
                    # Get the semantic ID at any pixel belonging to this instance
                    locations = np.where(instance_mask > 0)
                    if locations[0].size == 0:
                         print(f"Warning: Instance ID {inst_id} in {mask_filename} has no pixels. Skipping annotation.")
                         continue
                    target_semantic_id = int(semantic_channel[locations[0][0], locations[1][0]]) # Ensure it's a standard int

                    # Skip if semantic ID is background or not in categories list
                    if target_semantic_id == TARGET_BACKGROUND_ID or target_semantic_id not in sorted_target_ids:
                         # This instance belongs to a background or unmapped semantic category
                         # print(f"Info: Instance ID {inst_id} in {mask_filename} has target semantic ID {target_semantic_id}, which is background or unmapped. Skipping instance annotation.")
                         continue


                    # Convert mask to polygon
                    segmentation = mask_to_polygon(instance_mask)

                    if not segmentation:
                        # print(f"Warning: No valid polygon found for instance ID {inst_id} in {mask_filename}. Skipping annotation.")
                        continue

                    # Calculate area and bbox
                    area = float(np.sum(instance_mask)) # Area is pixel count
                    # Bbox: [x, y, width, height]
                    y_coords, x_coords = np.where(instance_mask > 0)
                    if y_coords.size > 0:
                        x_min, x_max = np.min(x_coords), np.max(x_coords)
                        y_min, y_max = np.min(y_coords), np.max(y_coords)
                        bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
                    else:
                        bbox = [0.0, 0.0, 0.0, 0.0] # Should not happen if segmentation is not empty

                    # Add annotation
                    annotation_id_counter += 1
                    coco_output["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": current_image_id,
                        "category_id": target_semantic_id,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0 # Instances are not crowd
                    })

                # --- Process Stuff (iscrowd=1) ---
                # Identify stuff pixels: instance_channel is 0 AND semantic_channel is not 0
                stuff_mask_all = (instance_channel == TARGET_BACKGROUND_ID) & (semantic_channel != TARGET_BACKGROUND_ID)

                # Find unique semantic IDs present in the stuff pixels
                unique_stuff_semantic_ids = np.unique(semantic_channel[stuff_mask_all])

                for target_semantic_id in unique_stuff_semantic_ids:
                     # Skip if semantic ID is background or not in categories list
                     if target_semantic_id == TARGET_BACKGROUND_ID or target_semantic_id not in sorted_target_ids:
                          # print(f"Info: Stuff region with target semantic ID {target_semantic_id} in {mask_filename} is background or unmapped. Skipping stuff annotation.")
                          continue

                     # Create a mask for the current stuff semantic region
                     current_stuff_semantic_mask = (semantic_channel == target_semantic_id) & stuff_mask_all

                     if np.sum(current_stuff_semantic_mask) == 0:
                          continue # Should be covered by unique_stuff_semantic_ids check, but safe

                     # Find connected components in this stuff region
                     num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(current_stuff_semantic_mask.astype(np.uint8), 8, cv2.CV_32S)

                     # Iterate through each connected component (excluding background label 0)
                     for i in range(1, num_labels):
                          component_mask = (labels_img == i).astype(np.uint8)

                          if np.sum(component_mask) == 0:
                               continue # Should not happen for labels 1 to num_labels-1

                          # Convert component mask to polygon
                          segmentation = mask_to_polygon(component_mask)

                          if not segmentation:
                               # print(f"Warning: No valid polygon found for stuff component {i} (semantic ID {target_semantic_id}) in {mask_filename}. Skipping annotation.")
                               continue

                          # Calculate area and bbox
                          area = float(np.sum(component_mask)) # Area is pixel count
                          # Bbox: [x, y, width, height]
                          y_coords, x_coords = np.where(component_mask > 0)
                          if y_coords.size > 0:
                              x_min, x_max = np.min(x_coords), np.max(x_coords)
                              y_min, y_max = np.min(y_coords), np.max(y_coords)
                              bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
                          else:
                              bbox = [0.0, 0.0, 0.0, 0.0] # Should not happen

                          # Add annotation
                          annotation_id_counter += 1
                          coco_output["annotations"].append({
                              "id": annotation_id_counter,
                              "image_id": current_image_id,
                              "category_id": target_semantic_id,
                              "segmentation": segmentation,
                              "area": area,
                              "bbox": bbox,
                              "iscrowd": 1 # Stuff is crowd
                          })


            except Exception as e:
                print(f"Error processing mask file {mask_filename} for COCO conversion: {e}. Skipping.")
                import traceback
                traceback.print_exc() # Print detailed error for debugging


        # Save the COCO JSON file
        try:
            with open(output_json_path, 'w') as f:
                json.dump(coco_output, f, indent=4)
            print(f"\nSuccessfully saved COCO JSON to {output_json_path}")
            print(f"Total images: {len(coco_output['images'])}")
            print(f"Total annotations: {len(coco_output['annotations'])}")
            print(f"Total categories: {len(coco_output['categories'])}")

        except IOError as e:
            print(f"Error saving COCO JSON file {output_json_path}: {e}")


# --- Main Execution ---

import sys # 导入 sys 模块来检查命令行参数

# --- Main Execution ---

if __name__ == "__main__":
    # 检查是否有命令行参数传入 (sys.argv[0] 是脚本名称本身)
    if len(sys.argv) > 1:
        print("Running with command-line arguments.")
        # 如果有参数，使用 argparse 解析
        parser = argparse.ArgumentParser(description="Convert instance/semantic segmentation annotations to a 3-channel mask format and optionally to COCO JSON.")
        parser.add_argument("--input_format", required=True, choices=['coco', 'separate_masks'],
                            help="Input annotation format ('coco' or 'separate_masks'). Add more choices as you implement parsers.")
        parser.add_argument("--input_dir", required=True,
                            help="Directory containing the original dataset (images and annotations).")
        parser.add_argument("--output_dir", required=True,
                            help="Directory to save the converted 3-channel mask images.")
        parser.add_argument("--category_mapping", default=CATEGORY_MAPPING_FILE,
                            help=f"Path to the category mapping JSON file (default: {CATEGORY_MAPPING_FILE}).")
        parser.add_argument("--image_subdir", default="images",
                            help="Subdirectory within input_dir containing image files (default: 'images').")
        parser.add_argument("--coco_annotation_file", default="annotations.json",
                            help="Annotation file name for COCO format (default: 'annotations.json').")
        parser.add_argument("--instance_mask_subdir", default="instance_masks",
                            help="Subdirectory within input_dir containing instance mask files for 'separate_masks' format (default: 'instance_masks').")
        parser.add_argument("--semantic_mask_subdir", default="semantic_masks",
                            help="Subdirectory within input_dir containing semantic mask files for 'separate_masks' format (default: 'semantic_masks').")
        parser.add_argument("--mask_ext", default=".png",
                            help="File extension for mask files in 'separate_masks' format (default: '.png').")
        parser.add_argument("--count_only", action="store_true",
                            help="Only count instances in existing masks in output_dir, do not perform conversion.")
        parser.add_argument("--count_dir", default=None,
                            help="Directory containing masks for counting (defaults to output_dir if count_only is used).")
        # --- New Argument for COCO JSON Output ---
        parser.add_argument("--output_coco_json", type=str, default=None,
                            help="Path to save the output COCO JSON annotation file. If provided, converts masks to COCO JSON.")
        parser.add_argument("--coco_mask_dir", type=str, default=None,
                            help="Directory containing the 3-channel masks to convert to COCO JSON (defaults to output_dir).")


        args = parser.parse_args()
    else:
        print("Running in IDE without command-line arguments. Using hardcoded defaults.")
        # 如果没有参数，手动设置默认值
        # 创建一个简单的对象来模拟 argparse.Namespace
        class IDEArgs:
            pass
        args = IDEArgs()

        # --- 在这里设置你的默认参数值 ---
        # !!! IMPORTANT: 请根据你的实际情况修改下面的路径和设置 !!!
        args.input_format = 'separate_masks' # 选择你要测试的格式: 'coco' 或 'separate_masks'
        args.input_dir = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/test' # 替换为你的测试数据集根目录
        args.output_dir = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/test/processed_masks_48_tmp'  # 替换为你希望保存转换结果的目录
        args.category_mapping = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/test/label2id_48.json' # 类别映射文件路径 (可以是相对或绝对路径)
        args.image_subdir = "images" # 图像文件所在的子目录

        # 针对不同格式的特定参数
        # 如果 input_format 是 'coco'
        # args.coco_annotation_file = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/coco_annotations_48.json" # COCO 标注文件相对于 input_dir 的路径

        # 如果 input_format 是 'separate_masks'
        args.instance_mask_subdir = "instance_masks" # 实例 mask 文件所在的子目录
        args.semantic_mask_subdir = "semantic_masks" # 语义 mask 文件所在的子目录
        args.mask_ext = ".png" # mask 文件的扩展名

        # 控制行为
        args.count_only = True  # 设置为 True 只计数，不转换
        args.count_dir = None  # 如果 count_only 为 True，可以指定计数目录，否则默认为 output_dir

        # --- 新增的 COCO JSON 输出参数 ---
        args.output_coco_json = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/COCO/pred.json' # 设置为 None 则不生成 COCO JSON
        args.coco_mask_dir = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/NYUv2/COCO/pred'  # 如果设置为 None，则使用 args.output_dir 作为 mask 来源

        # --- 默认参数设置结束 ---

        # 确保输出目录存在，即使使用硬编码路径
        os.makedirs(args.output_dir, exist_ok=True)
        # 确保 COCO JSON 输出目录存在
        if args.output_coco_json:
             output_json_dir = os.path.dirname(args.output_coco_json)
             if output_json_dir: # Check if path includes a directory
                 os.makedirs(output_json_dir, exist_ok=True)


    # 创建一个示例类别映射文件，如果它不存在的话
    # 这个逻辑无论参数来自命令行还是默认值都会执行
    if not os.path.exists(args.category_mapping):
        print(f"Creating a dummy category mapping file at {args.category_mapping}")
        print("!!! IMPORTANT: You MUST edit this file with your actual category mappings !!!")
        print("Target semantic IDs must be >= 1 for classes, and 0 for background.")
        dummy_mapping = {
            "example_source_instance_cat_1": 1,
            "example_source_instance_cat_2": 2,
            "example_source_semantic_cat_A": 3,
            "example_source_semantic_cat_B": 4,
            "background": 0,
            "unlabeled": 0,
            # 添加所有你的源类别 (名称或 ID) 及其对应的目标语义 ID
        }
        try:
            with open(args.category_mapping, 'w') as f:
                json.dump(dummy_mapping, f, indent=4)
        except IOError as e:
             print(f"Error creating dummy category mapping file {args.category_mapping}: {e}")
             # 如果无法创建映射文件，可能无法继续，这里可以选择退出或继续（可能会失败）
             # 为了健壮性，如果映射文件不存在且无法创建，最好退出
             sys.exit(f"Could not create dummy category mapping file at {args.category_mapping}. Please check permissions or create it manually.")


    try:
        # 使用 args 对象中的参数来初始化 Converter
        converter = AnnotationConverter(
            input_format=args.input_format,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            category_mapping_file=args.category_mapping,
            image_subdir=args.image_subdir
        )

        if args.count_only:
            # 只执行计数功能
            # count_dir_to_use = args.count_dir if args.count_dir is not None else args.output_dir
            # instance_counts = converter.count_instances_in_masks(mask_dir=count_dir_to_use)
            # print("Instance counts:", instance_counts) # Optional: print the returned dict

            # 如果同时请求了 COCO JSON 输出，则在计数后执行
            if args.output_coco_json:
                 coco_mask_dir_to_use = args.coco_mask_dir if args.coco_mask_dir is not None else args.output_dir
                 converter.convert_to_coco_json(args.output_coco_json, mask_dir=coco_mask_dir_to_use)

        else:
            # 执行转换功能
            # 根据 input_format 将特定参数传递给相应的解析方法
            # 使用 lambda 确保在调用 parse 方法时捕获 args 中的当前值
            if args.input_format == 'coco':
                converter.parser_map['coco'] = lambda: converter._parse_coco(annotation_file=args.coco_annotation_file)
            elif args.input_format == 'separate_masks':
                converter.parser_map['separate_masks'] = lambda: converter._parse_separate_masks(
                    instance_mask_subdir=args.instance_mask_subdir,
                    semantic_mask_subdir=args.semantic_mask_subdir,
                    mask_ext=args.mask_ext
                )

            # 执行转换
            converter.convert()

            # 转换完成后，执行计数和/或 COCO JSON 输出
            count_dir_to_use = args.count_dir if args.count_dir is not None else args.output_dir
            instance_counts = converter.count_instances_in_masks(mask_dir=count_dir_to_use)
            print("Instance counts:", instance_counts)  # Optional: print the returned dict

            if args.output_coco_json:
                 # 如果指定了 coco_mask_dir，则从该目录读取 mask；否则从 output_dir 读取
                 coco_mask_dir_to_use = args.coco_mask_dir if args.coco_mask_dir is not None else args.output_dir
                 converter.convert_to_coco_json(args.output_coco_json, mask_dir=coco_mask_dir_to_use)


    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the input directory, image subdirectory, annotation files/subdirectories, and category mapping file exist.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

