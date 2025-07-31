import tifffile
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import matplotlib.pyplot as plt
import export_coco
import json

def generate_distance_heatmap(mask_volume, distance_threshold_px_near=np.nan, distance_threshold_px_far=np.nan, 
                              near_color_rgb=(163, 222, 153), far_color_rgb=(205, 164, 224), object_color_rgb=(255,0,0), 
                              overlay=True, show=False, 
                              output_path=None, use_2d_xy_distances=False):
    """
    Create a 3D volume with pixels where pixels at least distance_threshold_px from the object instance represented in masks
    are colored.
    Inputs:
        mask_volume: np array of binary segmentation masks per frame.
        distance_threshold_px_near: Distance threshold in pixels. Pixels less than this distance from the object will be highlighted.
        distance_threshold_px_far: Distance threshold in pixels. Pixels more than this distance from the object will be highlighted.
        overlay: If True, returns a 3D volume where distant pixels are overlaid on the original masks.
                 If False, returns only the distance-based mask (highlighted pixels).
        show: if True, shows each slice.
        use_2d_xy_distances: if True, calculates distances in 2D within each slice only, not 3D across the volume.
    Outputs:
        A RGB 3D volume (Z,Y,X,3) where pixels meeting the distance condition are marked. If `overlay` is True, original object pixels are preserved.
    """
    # rgb replace the 1's in binary mask_volume with color.
    frames, h, w = mask_volume.shape
    output = np.zeros((frames, h, w, 3), dtype=np.uint8)

    inverted_mask_vol = np.logical_not(mask_volume).astype(np.uint8)
    # Distance transform: replaces each nonzero element with its shortest dist to zero elements
    if use_2d_xy_distances:
        # Compute distance per frame (XY only)
        distance_vol = np.zeros_like(mask_volume, dtype=np.float32)
        for i in range(mask_volume.shape[0]):
            distance_vol[i] = distance_transform_edt(inverted_mask_vol[i])
    else:
        # Use 3D distances
        distance_vol = distance_transform_edt(inverted_mask_vol)

    distance_mask_near = distance_vol <= distance_threshold_px_near
    distance_mask_far = distance_vol >= distance_threshold_px_far

    # Apply colors to output
    output[distance_mask_far] = far_color_rgb

    # Remove pixels near edge
    output[:,:distance_threshold_px_far, :] = np.array([0,0,0])
    output[:,:, :distance_threshold_px_far] = np.array([0,0,0])
    output[:,h-distance_threshold_px_far:, :] = np.array([0,0,0])
    output[:,:, w-distance_threshold_px_far:] = np.array([0,0,0])

    output[distance_mask_near] = near_color_rgb

    # Overlay object color where objects exist
    if overlay:
        output[mask_volume == 1] = object_color_rgb

    if show:
        for mask in output:
            plt.figure(figsize=(5, 5))
            plt.imshow(mask)
            plt.axis('off')
            plt.show()

    if output_path:
        tifffile.imwrite(
            output_path,
            output,
            bigtiff=True,
            photometric='rgb',
            compression='deflate' # Lossless
        )

    output = output.astype(np.uint8)
    return output

def regions_close_to_object_types(list_of_object_masks, thresh=50, color=[164, 222, 220]):
    """
    Inputs:
        list_of_object_masks: list of binary segmentation volumes: (Z,Y,X) np arrays, one for each individual object
            to include in this distance calculation. Must all be same shape.
        thresh (px): areas that are this distance or less away from all objects in list_of_object_masks
            will be highlighted.
    Returns:
        A colored RGB segmentation volume (Z,Y,X,3) in the same shape as the masks in list_of_object_masks. 
        Nonzero value means the pixel is less than thresh distance from all objects.
    """
    distance_vols = []
    frames, h, w = list_of_object_masks[0].shape
    output = np.zeros((frames, h, w, 3), dtype=np.uint8)

    for mask_volume in list_of_object_masks:

        # Get distances from all other areas to this object
        inverted_mask_vol = np.logical_not(mask_volume).astype(np.uint8)
        
        # Use 3D distances
        distance_vol = distance_transform_edt(inverted_mask_vol)    
        distance_vols.append(distance_vol)

        close_to_all = distance_vols[0]
        for i in range(len(distance_vols)-1):
            close_to_all = np.logical_and(close_to_all < thresh, distance_vols[i+1] < thresh).astype(np.uint8)
    
    output[close_to_all==1] = color
    output = output.astype(np.uint8)
    return output
def export_near_far_regions_as_coco(mask_volume, distance_threshold_px_near, distance_threshold_px_far,
                                     output_path="near_far_regions_coco.json", use_2d_xy_distances=False):
    """
    Generates COCO-format JSON for near and far regions around a binary mask.
    Only includes 'near_region' and 'far_region' categories in the final JSON.
    """

    from scipy.ndimage import distance_transform_edt
    import numpy as np
    import os

    inverted_mask_vol = np.logical_not(mask_volume).astype(np.uint8)
    if use_2d_xy_distances:
        distance_vol = np.zeros_like(mask_volume, dtype=np.float32)
        for i in range(mask_volume.shape[0]):
            distance_vol[i] = distance_transform_edt(inverted_mask_vol[i])
    else:
        distance_vol = distance_transform_edt(inverted_mask_vol)

    near_region_mask = (distance_vol <= distance_threshold_px_near).astype(np.uint8)
    far_region_mask  = (distance_vol >= distance_threshold_px_far).astype(np.uint8)

    # save json output
    mask_list = [near_region_mask, far_region_mask]
    export_coco.save_segmentations_as_coco(mask_list, coco_output_dir=output_path)

    # rename categories
    with open(output_path, "r") as f:
        coco_data = json.load(f)

    coco_data["categories"] = [
        {"id": 0, "name": "near_region", "supercategory": "distance"},
        {"id": 1, "name": "far_region", "supercategory": "distance"},
    ]

    valid_category_ids = {0, 1}
    coco_data["annotations"] = [
        ann for ann in coco_data["annotations"] if ann["category_id"] in valid_category_ids
    ]

    # overwrite original w renamed version
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)
