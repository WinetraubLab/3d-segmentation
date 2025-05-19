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

    def predict_mask(self, predictor, inference_state, frame_idx, prompt_mask, reverse=False):
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

    def _propagate_forward(self, images_to_segment_path, start_mask, start_keyframe_idx, end_keyframe_idx, class_id):
        """
        Forward pass for segmentation prediction given a start mask and end point.
        """
        predictor = self.initialize_segmentation_model()
        inference_state = predictor.init_state(video_path=images_to_segment_path, async_loading_frames=False)

        video_logits_f= {}
        video_segments_f = {}

        # add initial mask as conditioning frame
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_keyframe_idx,
            obj_id=class_id,
            mask=start_mask,
                )
        
        predicted_mask, predicted_logits = self.predict_mask(
            predictor,
            inference_state,
            start_keyframe_idx,
            {class_id: start_mask}
        )

        video_segments_f[start_keyframe_idx] = predicted_mask
        video_logits_f[start_keyframe_idx] = predicted_logits

        for out_frame_idx in range(start_keyframe_idx+1, end_keyframe_idx+1):
            predicted_mask, predicted_logits = self.predict_mask(predictor, inference_state, out_frame_idx, video_segments_f[out_frame_idx-1])

            video_segments_f[out_frame_idx] = predicted_mask
            video_logits_f[out_frame_idx] = predicted_logits

        return video_segments_f, video_logits_f
    
    def _propagate_reverse(self, images_to_segment_path, end_mask, start_keyframe_idx, end_keyframe_idx, class_id):
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

        # add initial mask as conditioning frame
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=end_keyframe_idx,
            obj_id=class_id,
            mask=end_mask,
                )

        predicted_mask, predicted_logits = self.predict_mask(
            predictor,
            inference_state,
            end_keyframe_idx,
            {class_id: end_mask},
            reverse=True
        )

        video_segments_b[end_keyframe_idx] = predicted_mask
        video_logits_b[end_keyframe_idx] = predicted_logits
        
        for out_frame_idx in reversed(range(start_keyframe_idx, end_keyframe_idx)):
            prompt_mask = video_segments_b[out_frame_idx + 1]
            predicted_mask, predicted_logits = self.predict_mask(
                predictor, inference_state, out_frame_idx, prompt_mask, reverse=True
            )

            video_segments_b[out_frame_idx] = predicted_mask
            video_logits_b[out_frame_idx] = predicted_logits

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
        global IMAGES_TO_SEGMENT_PATH
        IMAGES_TO_SEGMENT_PATH = import_data_from_roboflow.get_images_to_segment_path()

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
            segments, logits = self._propagate_forward(IMAGES_TO_SEGMENT_PATH, start_mask, start_idx, end_idx, class_id)
            video_segments_f.update(segments)
            video_logits_f.update(logits)

            end_mask = import_data_from_roboflow.get_mask(frame_names[end_idx], class_id)
            end_mask = np.array(start_mask).astype(np.uint8)

            # reverse
            segments, logits = self._propagate_reverse(IMAGES_TO_SEGMENT_PATH, end_mask, start_idx, end_idx, class_id)
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
                print(f"Using reverse masks for frame {t}")
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
