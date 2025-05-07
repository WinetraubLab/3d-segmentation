import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from collections import defaultdict

from pycocotools import mask as mask_utils
from torch.nn.functional import sigmoid

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask

def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def load_masks_from_dir(input_mask_path):
    input_mask, input_palette = load_ann_png(input_mask_path)
    per_obj_input_mask = get_per_obj_mask(input_mask)

    return per_obj_input_mask, input_palette


def create_mask_image(initial_mask_path, init_mask_idx, class_id, frame_names, COCO_PATH):
    """ 
    Convert COCO annotations to mask image
    """

    first_frame_name = frame_names[init_mask_idx] + ".jpg"
    os.makedirs(os.path.dirname(initial_mask_path), exist_ok=True)

    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    image_id_map = {img["file_name"]: img["id"] for img in coco["images"]}
    image_info_map = {img["id"]: img for img in coco["images"]}

    annotations_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)

    file_name = first_frame_name
    if file_name not in image_id_map:
        print(f"Skipping frame {file_name}, not in COCO annotations.")

    image_id = image_id_map[file_name]
    image_info = image_info_map[image_id]
    height, width = image_info["height"], image_info["width"]

    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    obj_id = 200
    for ann in annotations_per_image[image_id]:
        if ann.get("iscrowd", 0):
            rle = mask_utils.frPyObjects([ann["segmentation"]], height, width)
            m = mask_utils.decode(rle[0])
        else:
            rle = mask_utils.frPyObjects(ann["segmentation"], height, width)
            m = mask_utils.decode(rle)
            if len(m.shape) == 3:
                m = np.any(m, axis=2).astype(np.uint8)

        if ann["category_id"] == class_id: # tumor
            mask[m.astype(bool)] = obj_id

    Image.fromarray(mask).save(initial_mask_path)

    # plt.imshow(mask)
    # plt.show()

    # print(f"Saved mask for {first_frame_name} with {obj_id - 1} object(s)")
    # print(f"Unique values in mask: {np.unique(mask)}")


# combine class masks

def merge_bidirectional_masks(keyframe_indices, video_logits_f, video_logits_b, thresh=0.5):

    fused_masks = {}

    for t in range(keyframe_indices[0], keyframe_indices[1] + 1):
        if t in video_logits_f and t in video_logits_b:
            # print(f"Merging masks for frame {t}")
            frame_logits_f = video_logits_f[t]
            frame_logits_b = video_logits_b[t]
            fused_masks[t] = {}

            for obj_id in frame_logits_f.keys():
                if obj_id in frame_logits_b:
                    logit_f = frame_logits_f[obj_id]
                    logit_b = frame_logits_b[obj_id]

                    # Distance-weighted fusion: favor closer keyframe
                    alpha = (keyframe_indices[1] - t) / (keyframe_indices[1] - keyframe_indices[0])
                    fused_logit = alpha * logit_f + (1 - alpha) * logit_b

                    # Apply sigmoid to get probabilities
                    prob = sigmoid(fused_logit)

                    # Threshold to bool
                    mask = (prob > thresh).bool().numpy()

                    fused_masks[t][obj_id] = mask
        elif t in video_logits_f:
            print(f"Using forward masks for frame {t}")
            if t not in fused_masks:
                fused_masks[t] = {}
            for obj_id, logit in video_logits_f[t].items():
                fused_masks[t][obj_id] = (sigmoid(logit) > thresh).bool().numpy()
        elif t in video_logits_b:
            print(f"Using backward masks for frame {t}")
            if t not in fused_masks:
                fused_masks[t] = {}
            for obj_id, logit in video_logits_b[t].items():
                fused_masks[t][obj_id] = (sigmoid(logit) > thresh).bool().numpy()
        else:
            print(f"No masks found for frame {t}")

    return fused_masks


def get_file_map(directory):
    file_map = defaultdict(list)
    for fname in os.listdir(directory):
        name, ext = os.path.splitext(fname)
        file_map[name].append(os.path.join(directory, fname))
    return file_map

def combine_class_masks(mask_dirs, exclude_ranges=None, output_dir=None, show=True):
    if exclude_ranges is None:
        exclude_ranges = [["none", "none"] for _ in mask_dirs]

    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    mask_maps = []



    for i, mask_dir in enumerate(mask_dirs):
        if not os.path.exists(mask_dir):
            print(f"Skipping class {i+1} — directory not found: {mask_dir}")
            mask_maps.append({})
            continue

        mask_maps.append(get_file_map(mask_dir))

    shared_keys = set(mask_maps[0].keys())
    for d in mask_maps[1:]:
        shared_keys = sorted(set().union(*[d.keys() for d in mask_maps if d]))

    for base_name in shared_keys:
        masks = []
        bins = []
        try:
            number = int(base_name[1:5])
        except ValueError:
            print(f"Skipping invalid file name: {base_name}")
            continue

        for i in range(len(mask_dirs)):
            exclude = False
            start, end = exclude_ranges[i]
            if start != "none" and end != "none":
                if int(start) <= number <= int(end):
                    exclude = True

            if exclude:
                print(f"Excluding {base_name} for class {i}")
                # Create empty mask
                file_path = next(iter(mask_maps[i].values()))[0]
                empty_mask = np.zeros_like(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
                masks.append(empty_mask)
            elif base_name in mask_maps[i]:
                masks.append(cv2.imread(mask_maps[i][base_name][0], cv2.IMREAD_GRAYSCALE))
            else:
                ref_mask = masks[0] if masks else np.zeros((256, 256), dtype=np.uint8)  # empty if mask doesn't exist
                masks.append(np.zeros_like(ref_mask))

        for i in range(len(masks)):
            # Binarize masks
            bins.append((masks[i] > 0).astype(np.uint8))

        # Resolve conflicts: first class takes priority
        for i in range(len(bins)-1, 0, -1):
            bins[i][bins[i-1] == 1] = 0

        h, w = bins[0].shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        num_classes = len(bins)
        cmap = plt.get_cmap('Set3')
        colors = (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)

        for class_idx, binary_mask in enumerate(bins):
            color = colors[class_idx]
            rgb_mask[binary_mask == 1] = color

        if output_dir:
            out_path = os.path.join(output_dir, base_name + '.png')
            cv2.imwrite(out_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

        if show:
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb_mask)
            plt.title(base_name)
            plt.axis('off')
            plt.show()


