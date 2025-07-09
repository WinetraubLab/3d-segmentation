import tifffile
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import matplotlib.pyplot as plt

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

def regions_close_to_object_types(list_of_object_masks, thresh=50, use_2d_xy_distances=False):
    """
    Inputs:
        list_of_object_masks: list of binary segmentation volumes: (Z,Y,X) np arrays, one for each individual object
            to include in this distance calculation. Must all be same shape.
        thresh (px): areas that are this distance or less away from all objects in list_of_object_masks
            will be highlighted.
        use_2d_xy_distances: if True, calculates distance using only xy slices, not as a 3d volume.
    Returns:
        A binary segmentation volume (Z,Y,X) in the same shape as the masks in list_of_object_masks. 
        Value of 1 means the pixel is less than thresh distance from all objects.
    """
    distance_vols = []
    for mask_volume in list_of_object_masks:

        # Get distances from all other areas to this object
        inverted_mask_vol = np.logical_not(mask_volume).astype(np.uint8)
        if use_2d_xy_distances:
            # Compute distance per frame (XY only)
            distance_vol = np.zeros_like(mask_volume, dtype=np.float32)
            for i in range(mask_volume.shape[0]):
                distance_vol[i] = distance_transform_edt(inverted_mask_vol[i])
        else:
            # Use 3D distances
            distance_vol = distance_transform_edt(inverted_mask_vol)    
        distance_vols.append(distance_vol)

        close_to_all = distance_vols[0]
        for i in range(len(distance_vols)-1):
            close_to_all = np.logical_and(close_to_all < thresh, distance_vols[i+1] < thresh).astype(np.uint8)
    return close_to_all

def tumor_margin(tumor_segmentation_volume, distance_thresh_px=10):
    """
    Inputs:
        tumor_segmentation_volume: binary segmentation masks shape (Z,Y,X) where 1 denotes tumor
        distance_thresh_um: distance in px from boundary of tumor to consider part of the margin
    Returns: (Z,Y,X) binary mask volume
    """
    margin_mask = np.zeros_like(tumor_segmentation_volume)
    z_slices = tumor_segmentation_volume.shape[0]

    for z in range(z_slices):
        slice_ = tumor_segmentation_volume[z]

        # Find edges
        edge = ndimage.binary_dilation(slice_) ^ ndimage.binary_erosion(slice_)

        # Compute distance from border
        distance_map = ndimage.distance_transform_edt(~edge)

        # threshold
        within_mask = distance_map <= distance_thresh_px
        margin_mask[z] = within_mask

    return margin_mask

def tumor_core(tumor_segmentation_volume, distance_thresh_px):
    """
    Inputs:
        tumor_segmentation_volume: binary segmentation masks shape (Z,Y,X) where 1 denotes tumor
        distance_thresh_um: distance in px from boundary of tumor to consider part of the tumor core
    Returns: (Z,Y,X) binary mask volume
    """
    z_slices = tumor_segmentation_volume.shape[0]

    for z in range(z_slices):
        slice_ = tumor_segmentation_volume[z]

        # Compute distance from 0s
        distance_map = ndimage.distance_transform_edt(slice_)

        # threshold
        mask = distance_map >= distance_thresh_px

    return mask

def colorize_tumor(tumor_segmentation_mask, margin_distance_thresh_um, core_distance_thresh_um,
                   core_color_rgb=[252, 207, 3], margin_color_rgb=[0, 70, 140]):
    """
    Colorizes tumor based on distance classification (tumor margin or core).
    Inputs: 
        tumor_segmentation_mask: binary segmentation masks shape (Z,Y,X) where 1 denotes tumor
        margin_distance_thresh_um: distance in um from boundary of tumor to consider part of the margin
        core_distance_thresh_um: distance in um from boundary of tumor to consider part of the core
    Returns: (Z,Y,X,3) colored volume
    """
    dist_inside = distance_transform_edt(tumor_segmentation_mask)

    # Tumor core
    core_mask = (tumor_segmentation_mask == 1) & (dist_inside >= core_distance_thresh_um)

    # Tumor margin
    margin_mask = (tumor_segmentation_mask == 1) & (dist_inside <= margin_distance_thresh_um)

    # Intermediate regions
    intermediate_mask = (tumor_segmentation_mask == 1) & (~core_mask) & (~margin_mask)

    colored_volume = np.zeros(tumor_segmentation_mask.shape + (3,), dtype=np.uint8)
    gray         = [160, 160, 160] 
    colored_volume[core_mask] = core_color_rgb
    colored_volume[margin_mask] = margin_color_rgb
    colored_volume[intermediate_mask] = gray

    return colored_volume
