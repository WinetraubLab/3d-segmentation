from pycocotools import mask as mask_utils

import matplotlib.cm as cm
from collections import defaultdict
import tifffile
import json
import numpy as np

def convert_masks_to_coco(indiv_class_masks_list, num_frames, frame_names, image_height, image_width, coco_categories):
    """
    Converts a list of binary class masks into COCO annotations, using RLE encoding.
    Inputs:
        indiv_class_masks_list: list of np arrays; outer = per class, inner = per frame (2D array)
        num_frames: number of images segmented
        frame_names: list of the names of the images
        image_height, image_width: image dimensions. should be the same for all images in this stack.
        coco_categories: 'categories' dictionary entry from the original COCO file containing partial segmentations.
            Contains 'id' and 'name' information for each class.
    Returns:
        coco_output: dictionary in COCO format containing image metadata, predicted segmentations, and categories
    """
    annotations = []
    images = []
    annotation_id = 1

    for image_id in range(num_frames):
        images.append({
            "id": image_id,
            "file_name": frame_names[image_id], 
            "width": image_width,
            "height": image_height
        })

    for class_id, masks in enumerate(indiv_class_masks_list):
        for image_id, mask in enumerate(masks):
            if np.any(mask):
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii')  # for JSON serializability

                area = int(mask_utils.area(rle))
                bbox = list(map(float, mask_utils.toBbox(rle)))

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                annotation_id += 1
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories
    }
    return coco_output

def coco_to_tiff(coco_segmentations_path, tiff_output_path, colormap='Set3'):
    """
    Inputs:
        coco_segmentations_path: path to the COCO annotations file containing the propagated segmentations for all images.
        tiff_output_path: path and name of tiff file to save the output multi page tiff.
    """
    with open(coco_segmentations_path, "r") as f:
        coco = json.load(f)

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # initialize volume
    sorted_image_ids = sorted(annotations_by_image.keys())
    first_ann = annotations_by_image[sorted_image_ids[0]][0]
    height, width = first_ann['segmentation']['size']
    mask_volume = np.zeros((len(sorted_image_ids), height, width, 3), dtype=np.uint8)

    # set up colormap
    category_ids = sorted({ann['category_id'] for ann in coco['annotations']})
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    # Get colormap 
    cmap = cm.get_cmap(colormap, len(category_ids))
    color_list = (cmap(np.arange(len(category_ids)))[:, :3] * 255).astype(np.uint8)  # RGB

    # Visualize each image's mask
    for i, (image_id, anns) in enumerate(annotations_by_image.items()):
        mask_total = np.zeros((height,width,3), dtype=np.uint8)

        for ann in anns:
            seg = ann['segmentation']
            cat_id = ann['category_id']
            color = color_list[cat_id_to_index[cat_id]]

            # If counts is string, convert to bytes
            if isinstance(seg['counts'], str):
                seg['counts'] = seg['counts'].encode('utf-8')

            decoded_mask = mask_utils.decode(seg)

            # multi channel masks
            if decoded_mask.ndim == 3:
                mask = np.any(decoded_mask, axis=2)
            else:
                mask = decoded_mask

            for c in range(3):  # RGB 
                mask_total[:, :, c][mask == 1] = color[c]
                        
        mask_volume[i] = mask_total

    # Save to multi page tiff
    tifffile.imwrite(
        tiff_output_path,
        mask_volume,
        bigtiff=True,
        photometric='rgb',
        compression='deflate' # Lossless
    )