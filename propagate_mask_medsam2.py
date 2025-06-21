import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import torch
from scipy.ndimage import gaussian_filter

from MedSAM2.sam2.build_sam import build_sam2_video_predictor
import import_data_from_roboflow

class CustomMEDSAM2():
    def __init__(self, config_filepath, checkpoint_filepath):
        self.config_filepath = config_filepath
        self.checkpoint_filepath = checkpoint_filepath

    def initialize_new_predictor_state(self):
        """
        Initialize predictor for segmentation.
        Inputs:
            config_filepath: file path to configuration file.
            checkpoint_filepath: file path to pre-trained checkpoint to load.
        Returns:
            predictor: MEDSAM2 predictor
        """
        # Initialize predictor
        predictor = build_sam2_video_predictor(
            config_file= self.config_filepath,
            ckpt_path= self.checkpoint_filepath,
            apply_postprocessing=True,
            # hydra_overrides_extra=hydra_overrides_extra,
            vos_optimized=  True,
        )
        return predictor

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

    def _propagate_single_direction(self, class_id, reverse=False):
        """
        Forward or backward pass for segmentation prediction. Uses ground truth masks for keyframes, otherwise uses previous prediction.
        Inputs:
            class_id: class ID to segment
            reverse: if True, perform backward pass
        Returns:
            output_masks_binary: list of binary mask predictions. None if none
            output_masks_logit: list of logits for mask predictions. None if none
        """
        keyframe_indices = import_data_from_roboflow.get_keyframe_indices(class_id)
        global IMAGE_DATASET_FOLDER_PATH
        IMAGE_DATASET_FOLDER_PATH = import_data_from_roboflow.get_image_dataset_folder_path()

        # Initialize SAM model
        predictor = self.initialize_new_predictor_state()
        inference_state = predictor.init_state(video_path=IMAGE_DATASET_FOLDER_PATH, async_loading_frames=False)

        frame_names = import_data_from_roboflow.list_all_images()
        output_masks_logit = [None] * len(frame_names)
        output_masks_binary = [None] * len(frame_names)
        first_keyframe_idx = keyframe_indices[-1] if reverse else keyframe_indices[0]

        if not reverse:
            process_range = range(first_keyframe_idx, len(frame_names))
        else:
            process_range = range(first_keyframe_idx, -1, -1)

        for i in process_range:
            if i in keyframe_indices:
                gt_mask = np.array(import_data_from_roboflow.get_mask(frame_names[i], class_id)).astype(np.uint8)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=class_id,
                    mask=gt_mask,
                )
                predicted_mask = gt_mask
                predicted_logits = gt_mask
            else:
                prev_idx = process_range[i-1]
                if prev_idx < 0 or prev_idx >= len(frame_names) or output_masks_binary[prev_idx] is None:
                    predicted_mask = None
                    predicted_logits = None

                predicted_mask, predicted_logits = self._predict_mask(
                    predictor,
                    inference_state,
                    i,
                    output_masks_binary[prev_idx], 
                    class_id,
                    reverse=reverse
                )
                predicted_logits = predicted_logits.cpu().numpy()
                
            output_masks_binary[i] = np.squeeze(predicted_mask)
            output_masks_logit[i] = np.squeeze(predicted_logits)

        return output_masks_binary, output_masks_logit

    def propagate(self, class_id, sigma_xy=0, sigma_z=0):
        """
        Perform a global forward pass followed by a global backward pass, for specified class.
        Inputs:
            class_id: COCO class id to segment
        Returns:
            output_masks (dict): Predicted mask for each image
        """
        output_masks_binary_f, output_masks_logit_f = self._propagate_single_direction(class_id)
        output_masks_binary_b, output_masks_logit_b = self._propagate_single_direction(class_id, reverse=True)
        
        # Merge forward and backward predictions
        output_masks = self._merge_bidirectional_masks(output_masks_logit_f, output_masks_logit_b, class_id)
        output_masks = gaussian_filter(np.squeeze(np.array(output_masks)), sigma=(sigma_z, sigma_xy, sigma_xy))
        return output_masks
    
    def _merge_bidirectional_masks(self, output_masks_logit_f, output_masks_logit_b, class_id, thresh=0.5):
        """
        Merge forward and backward predictions using distance-weighted fusion between keyframe pairs.
        Inputs:
            output_masks_logit_f: list of 2D arrays or [] from forward pass
            output_masks_logit_b: list of 2D arrays or [] from backward pass
            class_id: class ID being segmented
            thresh: threshold for binary mask conversion
        Returns:
            output_masks: list of binary masks (numpy arrays) for each frame.
        """
        frame_names = import_data_from_roboflow.list_all_images()
        output_masks = [None] * len(frame_names)
        keyframe_indices = import_data_from_roboflow.get_keyframe_indices(class_id)

        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]

            for t in range(start_idx, end_idx + 1):
                logit_f = output_masks_logit_f[t] if output_masks_logit_f[t] is not None else None
                logit_b = output_masks_logit_b[t] if output_masks_logit_b[t] is not None else None

                if logit_f is not None and logit_b is not None:
                    alpha = (end_idx - t) / (end_idx - start_idx + 1e-5)  # avoid div by zero
                    logit_f = torch.tensor(logit_f)
                    logit_b = torch.tensor(logit_b)
                    fused_logit = alpha * logit_f + (1 - alpha) * logit_b
                elif logit_f is not None:
                    fused_logit = torch.tensor(logit_f)
                elif logit_b is not None:
                    fused_logit = torch.tensor(logit_b)
                else:
                    print(f"No logits found for frame {t}")
                    continue

                prob = torch.sigmoid(fused_logit)
                mask = (prob > thresh).bool().numpy()
                output_masks[t] = mask
                mask_shape = mask.shape

        output_masks = [
            np.full(mask_shape, np.nan) if mask is None else mask
            for mask in output_masks
        ]
        return output_masks

def combine_class_masks(indiv_class_masks_list, output_dir=None, show=True):
    """
    Combine multiple class masks into one RGB image per frame.
    Inputs:
        indiv_class_masks_list: list of lists; outer = per class, inner = per frame (2D array or None)
        output_dir: if set, saves masks as images to here
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
            mask = class_masks[frame_idx] if frame_idx < len(class_masks) else None

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
