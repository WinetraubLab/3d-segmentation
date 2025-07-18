from roboflow import Roboflow
from pycocotools import mask as mask_utils

import os
import cv2
import json
from collections import defaultdict
import numpy as np

COCO_PATH = ""                  # Path to the .coco annotation file.
IMAGE_DATASET_FOLDER_PATH = ""  # Path to the directory containing the images to segment.

def init_from_roboflow(workspace_name, project_name, api_key):
    """
    Downloads the project data from roboflow.

    Inputs:
        workspace_name: name of the Roboflow workspace.
        project_name: name of the Roboflow project.
        api_key: API key
    Returns:
        COCO_PATH: path to annotations file
        class_ids: a list of integers representing each unique class in the dataset segmentations
    """
    global COCO_PATH
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)

    # Automatically download the latest version
    versions = project.versions()
    latest_version = max(versions, key=lambda v: v.version)
    dataset = latest_version.download("coco-segmentation")

    for root, _, files in os.walk(dataset.location):
        for file in files:
            if file.endswith('.coco.json'):
                COCO_PATH = (os.path.join(root, file))

    class_ids = list_all_labels()

    return class_ids

def init_from_folder(folder_path):
    """
    Initialize from a folder containing COCO-style annotations. Sets global COCO_PATH: path to COCO annotation file.

    Inputs:
        folder_path: path to folder containing COCO-style annotation.
    Returns:
        class_ids: a list of integers representing each unique class in the dataset segmentations
    """
    global COCO_PATH
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.coco.json'):
                COCO_PATH = (os.path.join(root, file))
                 
    return list_all_labels()

def list_all_labels():
    """ 
    Returns:
        class_ids: list of integers representing each unique class in the segmentations.
    """
    global COCO_PATH
    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    # Get all class ids from segmentation annotations
    class_ids = {ann['category_id'] for ann in coco['annotations']}
    num_classes = len(class_ids)
    return class_ids

def list_all_images():
    """
    Inputs:
        folder_path: path to folder containing images to list
    Returns:
        image_names: list of filenames ending in png, jpg, jpeg, tif, or tiff, in sorted order.
    """
    global IMAGE_DATASET_FOLDER_PATH
    image_names = [
            p for p in os.listdir(IMAGE_DATASET_FOLDER_PATH)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF"]
        ]
    image_names = list(sorted(image_names))
    return image_names

def get_mask(image_name, class_id):
    """
    Inputs: 
        image_name: name of image
        class_id: integer representing the class to get the segmentation mask for
    Returns:
        mask: binary mask for the image and class
    """
    global COCO_PATH
    file_name = os.path.basename(image_name)

    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    image_id_map = {img["file_name"]: img["id"] for img in coco["images"]}
    image_info_map = {img["id"]: img for img in coco["images"]}
    
    def _normalize_filename(name):
        return ''.join(c for c in name if c.isalnum()).lower()

    matching_file = None
    for name in image_id_map:
        if _normalize_filename(name).startswith(_normalize_filename(file_name)):
            matching_file = name
            break

    if not matching_file:
        raise ValueError(f"No matching annotation found for image '{file_name}'.")

    annotations_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)

    image_id = image_id_map[matching_file]
    image_info = image_info_map[image_id]
    height, width = image_info["height"], image_info["width"]

    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations_per_image[image_id]:
        if ann["category_id"] != class_id:
            continue
        
        segmentation = ann["segmentation"]
        if ann.get("iscrowd", 0):
            rle = mask_utils.frPyObjects([segmentation], height, width)
            m = mask_utils.decode(rle[0])
        else:
            rle = mask_utils.frPyObjects(segmentation, height, width)
            m = mask_utils.decode(rle)
            if len(m.shape) == 3:
                m = np.any(m, axis=2).astype(np.uint8)
        
        mask[m.astype(bool)] = 1  # Binary mask

    return mask

def create_mask_volume(class_id, downsample_hw_size=None):
    """
    Inputs:
        class_id: coco-annotation class for which to create the segmentation mask volume.
    Returns:
        mask_volume: dictionary. dict[z] will be the segmentation mask loaded from COCO annotation file.
    """
    global IMAGE_DATASET_FOLDER_PATH

    image_files = list_all_images()
    mask_dict = {}

    for idx, image_f in enumerate(image_files):
        try:
            mask = get_mask(image_f, class_id).astype(np.float32)
            if downsample_hw_size is not None:
                assert len(downsample_hw_size) == 2
                mask = cv2.resize(mask, downsample_hw_size, interpolation=cv2.INTER_NEAREST) # nearest neighbor for masks
            mask_dict[idx] = mask
        except ValueError:
            continue  # Skip if no annotation

    return mask_dict

def preprocess_images(original_images_path, preprocessed_images_path, downsample_hw_size=None):
    """
    Apply CLAHE normalization and save new images.
    Inputs:
        original_images_path: directory containing original images.
        preprocessed_images_path: where to save preprocessed images.
        downsample_hw_size: (h,w) dimensions to resize the image to.
    """
    global IMAGE_DATASET_FOLDER_PATH
    IMAGE_DATASET_FOLDER_PATH = preprocessed_images_path
    os.makedirs(preprocessed_images_path, exist_ok=True)
    for root, _, files in os.walk(original_images_path):
        for file in files:
            if not file.lower().endswith('.json'):
                fpath = os.path.join(root, file)
                im = _clahe_normalize(cv2.imread(fpath))
                if downsample_hw_size is not None:
                    assert len(downsample_hw_size) == 2
                    if cv2.imread(fpath).ndim == 3:
                        oh, ow, _ = cv2.imread(fpath).shape
                    else:
                        oh, ow = cv2.imread(fpath).shape
                    nh, nw = downsample_hw_size
                    if nh > oh or nw > ow:
                        # the image is being upsampled rather than downsampled
                        im = cv2.resize(im, downsample_hw_size, interpolation=cv2.INTER_LINEAR) 
                    else:
                        im = cv2.resize(im, downsample_hw_size, interpolation=cv2.INTER_AREA) 
                
                cv2.imwrite(os.path.join(preprocessed_images_path, file), im)

def _clahe_normalize(image):
    """
    CLAHE normalization helps improve contrast for segmentation.
    Inputs:
        image: an image as array
    Returns:
        image_clahe: normalized image
    """

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Merge the channels and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return image_clahe

def get_image(image_name):
    """
    Inputs:
        image_name: full or partial image file name.
    Returns:
        image: the image with matching file name.
    """
    global IMAGE_DATASET_FOLDER_PATH
    for root, _, files in os.walk(IMAGE_DATASET_FOLDER_PATH):
        for file in files:
            if image_name in file:
                fpath = os.path.join(root, file)
                return _clahe_normalize(cv2.imread(fpath))
    
    raise FileNotFoundError(f"No matching images for {image_name} were found under {IMAGE_DATASET_FOLDER_PATH}.")

def get_class_id_from_name(class_name):
    global COCO_PATH
    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    # First, look for exact match
    for cat in coco['categories']:
        if cat['name'] == class_name:
            return cat['id']

    # Fallback to fuzzy match
    for cat in coco['categories']:
        if class_name in cat['name']:
            print(f"Warning: using fuzzy match for '{class_name}' → '{cat['name']}'")
            return cat['id']
        
    raise ValueError(f"Class name '{class_name}' not found in COCO categories.")

def get_all_class_name_id():
    global COCO_PATH
    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    # First, look for exact match
    for cat in coco['categories']:
        print(f"Class {cat['name']}: id {cat['id']}")

def get_image_dataset_folder_path():
    global IMAGE_DATASET_FOLDER_PATH
    return IMAGE_DATASET_FOLDER_PATH

def get_coco_path():
    global COCO_PATH
    return COCO_PATH
