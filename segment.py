import cv2
import numpy as np
from PIL import Image
import os
import json

from sam2.build_sam import build_sam2_video_predictor
from masks import put_per_obj_mask, load_masks_from_dir, create_mask_image, merge_bidirectional_masks

def save_predictions_to_dir(
    output_mask_dir,
    frame_name,
    per_obj_output_mask,
    height,
    width,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(output_mask_dir, exist_ok=True)

    output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
    output_mask_path = os.path.join(
        output_mask_dir, f"{frame_name}.png"
    )
    assert output_mask.dtype == np.uint8
    assert output_mask.ndim == 2
    output_mask = Image.fromarray(output_mask)
    output_mask.save(output_mask_path)

def save_clahe_images(input_path, clahe_path):
    # CLAHE normalization helps improve contrast for segmentation

    # Create output folder if it doesn't exist
    os.makedirs(clahe_path, exist_ok=True)

    for fn in os.listdir(input_path):
        if fn.lower().endswith('.jpg'):
            img = os.path.join(input_path, fn)

            output_path = os.path.join(clahe_path, fn)

            # Read the image
            image = cv2.imread(img)

            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to the L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)

            # Merge the channels and convert back to BGR
            lab_clahe = cv2.merge((l_clahe, a, b))
            image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Save the result
            cv2.imwrite(output_path, image_clahe)


def add_input_masks_from_keyframe(init_mask_idx, initial_mask_path, predictor, inference_state):
    """ 
    Prepare input masks for prompting.
    init_mask_idx: index of the mask file 
    """

    # Add input masks to MedSAM2 inference state before propagation
    object_ids_set = None
    input_frame_idx = init_mask_idx  # use first frame as mask input
    try:
        per_obj_input_mask, input_palette = load_masks_from_dir(input_mask_path=initial_mask_path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Failed to load input mask for frame {input_frame_idx=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        ) from e

    # get the list of object ids to track from the first input frame
    if object_ids_set is None:
        object_ids_set = set(per_obj_input_mask)
    for object_id, object_mask in per_obj_input_mask.items():
        # check and make sure no new object ids appear only in later frames
        if object_id not in object_ids_set:
            print(f"WARNING: Got no object ids on {input_frame_idx=}.")
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_id=object_id,
            mask=object_mask,
                )

    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        print(f"WARNING: Got no object ids on {input_frame_idx=}.")

    return object_ids_set

def run_bidirectional_inference_between_pairs(predictor, inference_state, pair_indices, class_id, frame_names, VIDEO_DIR, OCT_BASE_PATH, FORWARD_OUTDIR, BACKWARD_OUTDIR, COCO_PATH, forwards=True, backwards=True):
    """
    Run forward and backwards pass
    """
    start_idx, end_idx = pair_indices

    video_segments_f = {}
    video_segments_b = {}
    video_logits_f = {}
    video_logits_b = {}

    inference_state = predictor.init_state(video_path=VIDEO_DIR, async_loading_frames=False)

    height = inference_state["video_height"]
    width = inference_state["video_width"]

    print(f"\n=== FORWARD from frame {start_idx} to {end_idx} ===")

    create_mask_image(f"{OCT_BASE_PATH}/output_masks_class_{class_id}/" + frame_names[start_idx] + ".png", start_idx, class_id, frame_names, COCO_PATH)
    per_obj_input_mask, input_palette = load_masks_from_dir(f"{OCT_BASE_PATH}/output_masks_class_{class_id}/" + frame_names[start_idx] + ".png")

    for object_id, object_mask in per_obj_input_mask.items():
        # print(object_id)
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_idx,
            obj_id=object_id,
            mask=object_mask,
                )

    # if no objects, don't run in this direction
    if len(per_obj_input_mask) == 0:
        forwards = False

    if forwards:

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_idx,
            max_frame_num_to_track=end_idx-start_idx):
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

            # Save masks to disk
            save_predictions_to_dir(
                output_mask_dir=FORWARD_OUTDIR,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
            )

            # Feed each predicted mask as prompt for the next frame
            next_frame_idx = out_frame_idx + 1
            if next_frame_idx < end_idx:
                for obj_id, mask in per_obj_output_mask.items():
                    if mask is not None and np.any(mask):
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=next_frame_idx,
                            obj_id=obj_id,
                            mask=mask.squeeze(),
                        )

    # Reinitialize for backwards pass
    inference_state = predictor.init_state(
    video_path=VIDEO_DIR, async_loading_frames=False)

    print(f"\n=== BACKWARD from frame {end_idx} to {start_idx} ===")

    create_mask_image(f"{OCT_BASE_PATH}/output_masks_class_{class_id}/" + frame_names[end_idx] + ".png", end_idx, class_id, frame_names, COCO_PATH)
    per_obj_input_mask, input_palette = load_masks_from_dir(f"{OCT_BASE_PATH}/output_masks_class_{class_id}/" + frame_names[end_idx] + ".png")
    for object_id, object_mask in per_obj_input_mask.items():
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=end_idx,
            obj_id=object_id,
            mask=object_mask.squeeze(),
            )

    # if no objects, don't run in this direction
    if len(per_obj_input_mask) == 0:
        backwards = False

    if backwards:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=end_idx,
            max_frame_num_to_track=end_idx-start_idx, reverse=True):
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

            height = inference_state["video_height"]
            width = inference_state["video_width"]

            # Save masks to disk
            save_predictions_to_dir(
                output_mask_dir=BACKWARD_OUTDIR,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
            )

            # Feed each predicted mask as a prompt for the next frame
            next_frame_idx = out_frame_idx - 1

            if next_frame_idx > start_idx:
                for obj_id, mask in per_obj_output_mask.items():
                    if mask is not None and np.any(mask):
                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=next_frame_idx,
                            obj_id=obj_id,
                            mask=mask.squeeze(),
                        )
    return video_segments_f, video_segments_b, video_logits_f, video_logits_b, input_palette

def run_inference(orig_images_path, VIDEO_DIR, COCO_PATH, MODEL_CONFIG, MODEL_CHECKPOINT, OCT_BASE_PATH, FORWARD_OUTDIR, BACKWARD_OUTDIR):

    save_clahe_images(orig_images_path, VIDEO_DIR)

    # load the video frames
    frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(VIDEO_DIR)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    frame_names = list(sorted(frame_names))

    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    # Get all category_ids
    category_ids = {ann['category_id'] for ann in coco['annotations']}

    # Count unique classes
    num_classes = len(category_ids)

    print(f"Total number of unique classes: {num_classes}")

    for class_id in category_ids:
        ann_ids = sorted(list({ann["image_id"] for ann in coco["annotations"] if ann['category_id'] == class_id}))

        keyframe_filenames = [
            img['file_name'].split(".")[0] if 'file_name' in img else img['name']
            for img in coco['images']
            if img['id'] in ann_ids
        ]

        keyframe_indices = {0, len(frame_names)-1}
        for ann in keyframe_filenames:
            for i, name in enumerate(frame_names):
                if ann in name:
                    keyframe_indices.add(i)

        keyframe_indices = sorted(list(keyframe_indices))

        for i in range(len(keyframe_indices)-1):

            # Initialize predictor
            predictor = build_sam2_video_predictor(
                config_file=MODEL_CONFIG,
                ckpt_path=MODEL_CHECKPOINT,
                apply_postprocessing=True,
                # hydra_overrides_extra=hydra_overrides_extra,
                vos_optimized=  True,
            )
            inference_state = predictor.init_state(video_path=VIDEO_DIR, async_loading_frames=False)

            kf_pair = [keyframe_indices[i], keyframe_indices[i+1]]
            print("Evaluating pair: ", kf_pair)

            matching_frame_name = next((name for name in frame_names if keyframe_filenames[i] in name), None)
            initial_mask_path = f"{OCT_BASE_PATH}/output_masks_class_{class_id}/{matching_frame_name}.png"

            # Create masks from COCO
            mask = create_mask_image(initial_mask_path, kf_pair[0], class_id, frame_names, COCO_PATH)
            add_input_masks_from_keyframe(kf_pair[0], initial_mask_path, predictor, inference_state)

            video_segments_f, video_segments_b, video_logits_f, video_logits_b, input_palette = run_bidirectional_inference_between_pairs(predictor, inference_state, kf_pair, class_id, frame_names, VIDEO_DIR, OCT_BASE_PATH, FORWARD_OUTDIR, BACKWARD_OUTDIR)

            fused_masks = merge_bidirectional_masks(kf_pair, video_logits_f, video_logits_b)

            height = inference_state["video_height"]
            width = inference_state["video_width"]

            for t, obj_masks in fused_masks.items():
                    save_predictions_to_dir(
                        output_mask_dir=f"{OCT_BASE_PATH}/merged_predictions_class_{class_id}/",
                        frame_name=frame_names[t],
                        per_obj_output_mask=obj_masks,
                        height=height,
                        width=width,
                    )
                    
    return category_ids
