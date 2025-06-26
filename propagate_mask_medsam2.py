import numpy as np
import os
import cv2

import json
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import torch
from scipy.ndimage import gaussian_filter

from MedSAM2.sam2.build_sam import build_sam2_video_predictor
import import_data_from_roboflow
import export_coco

class CustomMEDSAM2():
    def __init__(self, config_filepath, checkpoint_filepath):
        self.config_filepath = config_filepath
        self.checkpoint_filepath = checkpoint_filepath

    def _predict_mask(self, predictor, inference_state, frame_idx, prompt_mask, class_id, reverse=False):
        """
        Given an image and the input prompting mask, predict the mask for the image.
        Inputs:
            predictor: segmentation model instance
            inference_state: current tracking/memory state
            frame_idx: int index of frame to predict
            prompt_mask: prompting mask. from previous image frame_idx - 1
        """
        # Set up tracking
        if prompt_mask is not None and np.any(prompt_mask):
            # Tell model that this frame has already been tracked = refine mask mode
            if frame_idx not in inference_state["frames_already_tracked"]:
                inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=class_id,
                mask=prompt_mask.squeeze(),
            )

        # Predict
        for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx,
            max_frame_num_to_track=1, reverse=reverse):

            for i, obj_id in enumerate(out_obj_ids):
                if obj_id == class_id:
                    logit = out_mask_logits[i].cpu()
                    mask = (logit > 0.0).numpy()
                    return mask, logit

        return None, None 
    
    def _get_keyframe_indices_from_sparse_segmentations(self, binary_segmentations):
        """
        Inputs:
            binary_segmentations: array containing binary segmentation mask for some frames. if no binary segmentation mask for a certain
                frame, it will be NaN
        Returns:
            keyframe_indices: list of indices where there is a valid seg mask
        """
        all_nan = np.isnan(binary_segmentations).all(axis=(1, 2))
        not_all_nan = ~all_nan
        keyframe_indices = np.where(not_all_nan)[0]
        return keyframe_indices

    def _propagate_single_direction(self, image_dataset_folder_path, binary_segmentations, reverse=False):
        """
        Forward or backward pass for segmentation prediction. Uses ground truth masks for keyframes, otherwise uses previous prediction.
        Inputs:
            image_dataset_folder_path: Directory containing preprocessed images to segment.
            binary_segmentations: array containing binary segmentation mask for some frames. if no binary segmentation mask for a certain
                frame, it will be NaN
            reverse: if True, perform backward pass
        Returns:
            output_masks_binary: list of binary mask predictions (ndarray). NaN for a specific frame if no valid prediction 
            output_masks_logit: list of logits for mask predictions (ndarray). NaN for a specific frame if no valid prediction 
        """
        # get valid segmentation indices
        keyframe_indices = self._get_keyframe_indices_from_sparse_segmentations(binary_segmentations)

        # Initialize SAM model
        predictor = build_sam2_video_predictor(
            config_file= self.config_filepath,
            ckpt_path= self.checkpoint_filepath,
            apply_postprocessing=True,
            # hydra_overrides_extra=hydra_overrides_extra,
            vos_optimized=  True,
        )
        inference_state = predictor.init_state(video_path=image_dataset_folder_path, async_loading_frames=False)
        mask_shape = binary_segmentations[keyframe_indices[0]].shape

        image_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF"]
        n_frames =  sum(
            1 for filename in os.listdir(image_dataset_folder_path)
            if os.path.isfile(os.path.join(image_dataset_folder_path, filename)) and os.path.splitext(filename)[1] in image_extensions
        )

        output_masks_logit = [np.full(mask_shape, np.nan) for _ in range(n_frames)]
        output_masks_binary = [np.full(mask_shape, np.nan) for _ in range(n_frames)]
        first_keyframe_idx = keyframe_indices[-1] if reverse else keyframe_indices[0]

        # range of valid indices to propagate segmentations through
        if not reverse:
            process_range = range(first_keyframe_idx, n_frames)
        else:
            process_range = range(first_keyframe_idx, -1, -1)

        for i in process_range:
            if i in keyframe_indices:
                # if mask segmentation is known, set mask and logits
                gt_mask = binary_segmentations[i]
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=0,
                    mask=gt_mask,
                )
                predicted_mask = gt_mask
                predicted_logits = (gt_mask * 20.0) - 10.0  # large positive where mask=1, large neg where mask=0
            else:
                # otherwise, predict current mask using previous frame
                prev_idx = process_range[i-1]
                if prev_idx < 0 or prev_idx >= n_frames or output_masks_binary[prev_idx] is None:
                    predicted_mask = np.full(mask_shape, np.nan)
                    predicted_logits = np.full(mask_shape, np.nan)

                predicted_mask, predicted_logits = self._predict_mask(
                    predictor,
                    inference_state,
                    i,
                    output_masks_binary[prev_idx], 
                    0,
                    reverse=reverse
                )
                predicted_logits = predicted_logits.cpu().numpy()
                
            output_masks_binary[i] = np.squeeze(predicted_mask)
            output_masks_logit[i] = np.squeeze(predicted_logits)

        return output_masks_binary, output_masks_logit

    def propagate(self, image_dataset_folder_path, binary_segmentations, sigma_xy=0.0, sigma_z=0.0, prob_thresh=0.5):
        """
        This function will initialize a model from sparse segmentation and propagate the segmentation to all frames.
        Inputs:
            image_dataset_folder_path: Directory containing preprocessed images to segment.
            binary_segmentations: array of masks, with each element corresponding to a different frame in a 3D volume. 
                Elements in the array can either be NaN, indicating that no mask is provided for the specific frame (meaning we need to propagate to it), 
                or a numpy array of shape (n, m), which specifies whether each pixel in the frame (with dimensions n by m) is inside (1) or outside (0) the mask (key frame).
                For example, binary_segmentations = [NaN, zeros(n,m), NaN, NaN, zeros(n,m)] defines key frames in frames 1 and 4.
            sigma_xy: gaussian smoothing sigma on x and y axes
            sigma_z: gaussian smoothing sigma on z axis
            prob_thresh: probability threshold; values above this are set to 1 in the binary mask.
        Returns:
            output_masks: 3D numpy matrix shape (z,x,y) that for each pixel defines if it's inside (1) or outside (0) a mask. 
                Elements in array will be NaN for slices with no predictions.
        """
        mask_binary_forward, mask_logit_forward = self._propagate_single_direction(image_dataset_folder_path, binary_segmentations)
        mask_binary_backward, mask_logit_backward = self._propagate_single_direction(image_dataset_folder_path, binary_segmentations, reverse=True)
        
        # Merge forward and backward predictions
        avg_logits = torch.tensor(np.nanmean(np.stack([mask_logit_forward, mask_logit_backward]), axis=0))
        prob = torch.sigmoid(avg_logits)

        smoothed_probs = gaussian_filter(np.squeeze(np.array(prob)), sigma=(sigma_z, sigma_xy, sigma_xy))
        # Scale to preserve maximums
        smoothed_probs *= prob.max().item()/smoothed_probs.max()
        masks = (smoothed_probs > prob_thresh).astype(bool)
        return masks

def combine_class_masks(indiv_class_masks_list, output_dir=None, output_as_coco=True, coco_output_dir="predicted_segmentations_coco.json", show=True):
    """
    Combine multiple class masks into one RGB image per frame.
    Inputs:
        indiv_class_masks_list: list of np arrays; outer = per class, inner = per frame (2D array)
        output_dir: if set, saves masks as images to here
        output_as_coco: if True, writes segmentations to a file in COCO annotation format
        coco_output_dir: location to write the final segmentations in COCO format
        show: if True, displays combined masks
    """
    frame_names = import_data_from_roboflow.list_all_images()
    num_classes = len(indiv_class_masks_list)
    num_frames = len(frame_names)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for frame_idx in range(num_frames):
        masks = []
        h, w = None, None

        # Gather all masks for this frame
        for class_idx in range(num_classes):
            class_masks = indiv_class_masks_list[class_idx]
            mask = class_masks[frame_idx] if frame_idx < class_masks.shape[0] else None

            if mask is not None:
                mask = np.array(mask).squeeze()
                if h is None or w is None:
                    h, w = mask.shape
            masks.append(mask)

        if h is None or w is None:
            print(f"Skipping frame {frame_idx}: no valid masks.")
            continue

        # Binarize masks
        bins = []
        for m in masks:
            if m is not None:
                bins.append((m > 0).astype(np.uint8))
            else:
                bins.append(np.zeros((h, w), dtype=np.uint8))

        # Enforce class priority
        for i in range(len(bins) - 1, 0, -1):
            bins[i][bins[i - 1] == 1] = 0

        # Assign colors
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cmap = plt.get_cmap('Set3')
        colors = (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)

        for class_idx, binary_mask in enumerate(bins):
            rgb_mask[binary_mask == 1] = colors[class_idx]

        # Save image
        if output_dir:
            out_path = os.path.join(output_dir, f"{frame_names[frame_idx]}.png")
            cv2.imwrite(out_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

        if show:
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb_mask)
            plt.title(frame_names[frame_idx])
            plt.axis('off')
            plt.show()
    
    if output_as_coco:
        with open(import_data_from_roboflow.COCO_PATH, "r") as f:
            coco_data = json.load(f)
        coco_categories = coco_data.get("categories", [])
        coco_output = export_coco.convert_masks_to_coco(indiv_class_masks_list, num_frames, frame_names, h, w, coco_categories)
        with open(coco_output_dir, "w") as f:
            json.dump(coco_output, f, indent=4)
    return coco_output
