import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid

from MedSAM2.sam2.build_sam import build_sam2_video_predictor
import import_data_from_roboflow

class CustomMEDSAM2():
    def __init__(self, config_filepath, checkpoint_filepath):
        self.config_filepath = config_filepath
        self.checkpoint_filepath = checkpoint_filepath

    def initialize_segmentation_model(self):
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
    
    def _propagate_forward(self, images_to_segment_path, start_mask, start_keyframe_idx, end_keyframe_idx, class_id):
        """
        Forward pass for segmentation prediction given a start mask and end point.
        """
        predictor = self.initialize_segmentation_model()
        inference_state = predictor.init_state(video_path=images_to_segment_path, async_loading_frames=False)

        video_logits_f= {}
        video_segments_f = {}

        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_keyframe_idx,
            obj_id=class_id,
            mask=start_mask,
                )
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_keyframe_idx,
            max_frame_num_to_track=end_keyframe_idx-start_keyframe_idx):
            # Get binary masks
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Save segmentation
            video_segments_f[out_frame_idx] = per_obj_output_mask

            # Store logits
            video_logits_f[out_frame_idx] = {
                out_obj_id: out_mask_logits[i].cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Feed each predicted mask as prompt for the next frame
            next_frame_idx = out_frame_idx + 1
            if next_frame_idx < end_keyframe_idx:
                for obj_id, mask in per_obj_output_mask.items():
                    if mask is not None and np.any(mask):
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=next_frame_idx,
                            obj_id=obj_id,
                            mask=mask.squeeze(),
                        )
        return video_segments_f, video_logits_f

    
    def _propagate_backward(self, images_to_segment_path, end_mask,  start_keyframe_idx, end_keyframe_idx, class_id):
        """
        Backwards pass for segmentation prediction given a start mask and end point.
        """
        predictor = self.initialize_segmentation_model()
        inference_state = predictor.init_state(video_path=images_to_segment_path, async_loading_frames=False)

        video_logits_b= {}
        video_segments_b = {}

        # switch start and end idx if start is greater than end
        if start_keyframe_idx > end_keyframe_idx:
            tmp = end_keyframe_idx
            end_keyframe_idx = start_keyframe_idx
            start_keyframe_idx = tmp

        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=end_keyframe_idx,
            obj_id=class_id,
            mask=end_mask,
                )
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=end_keyframe_idx,
            max_frame_num_to_track=end_keyframe_idx-start_keyframe_idx, reverse=True):
            # Get binary masks
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Save segmentation
            video_segments_b[out_frame_idx] = per_obj_output_mask

            # Store logits
            video_logits_b[out_frame_idx] = {
                out_obj_id: out_mask_logits[i].cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Feed each predicted mask as prompt for the next frame
            next_frame_idx = out_frame_idx - 1

            if next_frame_idx < end_keyframe_idx:
                for obj_id, mask in per_obj_output_mask.items():
                    if mask is not None and np.any(mask):
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=next_frame_idx,
                            obj_id=obj_id,
                            mask=mask.squeeze(),
                        )
        return video_segments_b, video_logits_b


    def propagate_sequence(self, class_id):
        """
        Propagate masks from start_idx to end_idx (inclusive) using MedSAM2, for specified class.
        Inputs:
            images_to_segment_path: path to directory of image stack
            class_id: COCO category id for segmentation
            coco_path: COCO annotation file path 
        Returns:
            fused_masks (dict): Predicted mask for each image
            frame_names (list): list of image file names, in order
        """
        images_to_segment_path = import_data_from_roboflow.IMAGES_TO_SEGMENT_PATH

        frame_names = import_data_from_roboflow.list_all_images()
        keyframe_indices = import_data_from_roboflow.get_keyframe_indices(class_id)
        num_frames = len(frame_names)

        start_idx = 0
        end_idx = num_frames - 1

        video_segments_f = {}
        video_logits_f = {}
        video_segments_b = {}
        video_logits_b = {}
        fused_masks = {}

        for i in range(0, len(keyframe_indices)-1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i+1]

            start_mask = import_data_from_roboflow.get_mask(frame_names[start_idx], class_id)
            start_mask = np.array(start_mask).astype(np.uint8)

            # forward pass
            segments, logits = self._propagate_forward(images_to_segment_path, start_mask, start_idx, end_idx, class_id)
            video_segments_f.update(segments)
            video_logits_f.update(logits)

            end_mask = import_data_from_roboflow.get_mask(frame_names[end_idx], class_id)
            end_mask = np.array(start_mask).astype(np.uint8)

            # backwards
            segments, logits = self._propagate_backward(images_to_segment_path, end_mask, start_idx, end_idx, class_id)
            video_segments_b.update(segments)
            video_logits_b.update(logits)

            fused_masks.update(self.merge_bidirectional_masks([start_idx, end_idx], video_logits_f, video_logits_b))

        return fused_masks, frame_names


    def merge_bidirectional_masks(self, keyframe_indices, video_logits_f, video_logits_b, thresh=0.5):

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

def combine_class_masks(indiv_class_masks_list, frame_names, output_dir=None, show=True):
    """
    Combine multiple class masks into one mask. Different colors represent different classes.
    Inputs:
        indiv_class_masks_list: list containing a dictionary of masks for each of n classes segmented
        output_dir: directory to write the output images to
        show: if True, show combined class masks
    """

    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    shared_keys = set()
    for d in indiv_class_masks_list:
        shared_keys.update(d.keys())
    shared_keys = sorted(shared_keys)

    for frame_idx in shared_keys:
        masks = []
        bins = []

        for i, class_mask_dict in enumerate(indiv_class_masks_list):
            masks.append(class_mask_dict.get(frame_idx, np.zeros_like(next(iter(class_mask_dict.values())))))

        for i in range(len(masks)):
            # Binarize masks
            bins.append((list(masks[i].values())[0] > 0).astype(np.uint8).squeeze())

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
            out_path = os.path.join(output_dir, frame_names[frame_idx] + '.png')
            cv2.imwrite(out_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

        if show:
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb_mask)
            plt.title(frame_names[frame_idx])
            plt.axis('off')
            plt.show()
