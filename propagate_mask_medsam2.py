import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import torch

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

    def _predict_mask(self, predictor, inference_state, frame_idx, prompt_mask, reverse=False):
        """
        Given an image and the input prompting mask, predict the mask for the image.
        Inputs:
            predictor: segmentation model instance
            inference_state: current tracking/memory state
            frame_idx: int index of frame to predict
            prompt_mask: prompting mask. dict of {obj_id: binary mask} from previous image frame_idx - 1
        """
        predicted_logits = {}
        predicted_masks = {}
        # Set up tracking
        for obj_id, mask in prompt_mask.items():
            if mask is not None and np.any(mask):
                # Tell model that this frame has already been tracked = refine mask mode
                if frame_idx not in inference_state["frames_already_tracked"]:
                    inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask.squeeze(),
                )

        # Predict
        for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_idx,
            max_frame_num_to_track=1, reverse=reverse):

            predicted_masks = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            predicted_logits = {
                out_obj_id: out_mask_logits[i].cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return predicted_masks, predicted_logits

    def _propagate_single_direction(self, class_id, reverse=False):
        """
        Forward or backward pass for segmentation prediction. Uses ground truth masks for keyframes, otherwise uses previous prediction.
        Inputs:
            class_id: class ID to segment
            reverse: if True, perform backward pass
        Returns:
            output_masks_binary: pass predictions
            output_masks_logit: pass logits
        """
        keyframe_indices = import_data_from_roboflow.get_keyframe_indices(class_id)
        global IMAGE_DATASET_FOLDER_PATH
        IMAGE_DATASET_FOLDER_PATH = import_data_from_roboflow.get_image_dataset_folder_path()

        # Initialize SAM model
        predictor = self.initialize_new_predictor_state()
        inference_state = predictor.init_state(video_path=IMAGE_DATASET_FOLDER_PATH, async_loading_frames=False)

        output_masks_logit = {}
        output_masks_binary = {}
        
        frame_names = import_data_from_roboflow.list_all_images()

        # First handle frames before first keyframe - no predictions
        first_keyframe_idx = keyframe_indices[::-1] if reverse else keyframe_indices[0]
        if not reverse:
            skip_range = range(0, first_keyframe_idx)
        else:
            skip_range = range(first_keyframe_idx, len(frame_names))
        
        for frame_idx in skip_range:
            output_masks_binary[frame_idx] = {class_id: None}
            output_masks_logit[frame_idx] = {class_id: None}

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
                predicted_mask = {class_id: gt_mask}
                predicted_logits = {class_id: gt_mask}
            else:
                prev_idx = i - 1 if not reverse else i + 1
                predicted_mask, predicted_logits = self._predict_mask(
                    predictor,
                    inference_state,
                    i,
                    output_masks_binary[prev_idx],
                    reverse=reverse
                )
            output_masks_binary[i] = predicted_mask
            output_masks_logit[i] = predicted_logits

        return output_masks_binary, output_masks_logit

    def propagate(self, class_id):
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
        return output_masks

    def _merge_bidirectional_masks(self, output_masks_logit_f, output_masks_logit_b, class_id, thresh=0.5):
        """
        Merge forward and backward predictions using distance-weighted fusion between keyframe pairs.
        Inputs:
            output_masks_logit_f: forward pass logits for all frames
            output_masks_logit_b: backward pass logits for all frames
            class_id: class ID being segmented
            thresh: threshold for binary mask conversion
        Returns:
            output_masks: dictionary of merged masks for each frame
        """
        output_masks = {}
        keyframe_indices = import_data_from_roboflow.get_keyframe_indices(class_id)

        # For each pair of consecutive keyframes
        for i in range(len(keyframe_indices)-1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i+1]
            
            # Merge masks for frames between this keyframe pair
            for t in range(start_idx, end_idx + 1):
                if t in output_masks_logit_f and t in output_masks_logit_b:
                    frame_logits_f = output_masks_logit_f[t]
                    frame_logits_b = output_masks_logit_b[t]
                    output_masks[t] = {}

                    for obj_id in frame_logits_f.keys():
                        if obj_id in frame_logits_b:
                            logit_f = frame_logits_f[obj_id]
                            logit_b = frame_logits_b[obj_id]

                            # Distance-weighted fusion: favor closer keyframe
                            alpha = (end_idx - t) / (end_idx - start_idx)
                            fused_logit = alpha * logit_f + (1 - alpha) * logit_b

                            # Apply sigmoid to get probabilities
                            if not isinstance(fused_logit, torch.Tensor):
                                fused_logit = torch.tensor(fused_logit)
                            prob = sigmoid(fused_logit)

                            # Threshold to bool
                            mask = (prob > thresh).bool().numpy()

                            output_masks[t][obj_id] = mask
                        else:
                            logit_f = frame_logits_f[obj_id]
                            fused_logit = logit_f
                            prob = sigmoid(fused_logit)
                            mask = (prob > thresh).bool().numpy()
                            output_masks[t][obj_id] = mask
                elif t in output_masks_logit_f:
                    print(f"Using forward masks for frame {t}")
                    if t not in output_masks:
                        output_masks[t] = {}
                    for logit in output_masks_logit_f[t]:
                        if not isinstance(logit, torch.Tensor):
                            logit = torch.tensor(logit)
                        output_masks[t] = (sigmoid(logit) > thresh).bool().numpy()
                elif t in output_masks_logit_b:
                    print(f"Using reverse masks for frame {t}")
                    if t not in output_masks:
                        output_masks[t] = {}
                    for logit in output_masks_logit_b[t]:
                        if not isinstance(logit, torch.Tensor):
                            logit = torch.tensor(logit)
                        output_masks[t] = (sigmoid(logit) > thresh).bool().numpy()
                else:
                    print(f"No masks found for frame {t}")

        return output_masks

def combine_class_masks(indiv_class_masks_list, output_dir=None, show=True):
    """
    Combine multiple class masks into one mask. Different colors represent different classes.
    Inputs:
        indiv_class_masks_list: list containing a dictionary of masks for each of n classes segmented
        output_dir: directory to write the output images to
        show: if True, show combined class masks
    """
    frame_names = import_data_from_roboflow.list_all_images()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_frame_indices = set()
    for class_dict in indiv_class_masks_list:
        all_frame_indices.update(class_dict.keys())
    all_frame_indices = sorted(all_frame_indices)

    for frame_idx in all_frame_indices:
        masks = []
        bins = []

        # get binary masks for each class 
        for class_mask_dict in indiv_class_masks_list:
            class_masks = class_mask_dict.get(frame_idx, {})
            if class_masks:
                mask_array = list(class_masks.values())[0]
                masks.append(mask_array)
            else:
                masks.append(None)

        h, w = None, None
        for m in masks:
            if m is not None:
                m = m.squeeze()
                h, w = m.shape
                break
        if h is None or w is None:
            print(f"Skipping frame {frame_idx}: no masks available.")
            continue

        #  fill masks
        for m in masks:
            if m is not None:
                bins.append((m > 0).astype(np.uint8).squeeze())
            else:
                bins.append(np.zeros((h, w), dtype=np.uint8))

        # first class takes priority
        for i in range(len(bins) - 1, 0, -1):
            bins[i][bins[i - 1] == 1] = 0

        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        num_classes = len(bins)
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
