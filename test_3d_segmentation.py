import numpy as np
import numpy.testing as npt
import unittest
import import_data_from_roboflow, propagate_mask_medsam2, export_coco, analyze_volumes

class Test3dSegmentation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.original_images_path = "test_vectors/sample_image_sequence/"
        self.image_dataset_folder_path = "test_vectors/preprocessed_images/"
        self.workspace_name = "yolab-kmmfx"  
        self.project_name = "vol1_2"
        self.api_key = "" # Set when testing test_init_roboflow()
        self.MODEL_CONFIG = "configs/sam2.1_hiera_t512.yaml"
        self.MODEL_CHECKPOINT = "MedSAM2/checkpoints/MedSAM2_latest.pt"

    def set_up_global_vars(self):
        import_data_from_roboflow.init_from_folder("test_vectors/sample_annotations_folder/")
        import_data_from_roboflow.preprocess_images(self.original_images_path, self.image_dataset_folder_path)
        self.seg_vol_0 = import_data_from_roboflow.create_mask_volume(0)
        self.seg_vol_1 = import_data_from_roboflow.create_mask_volume(1)
        
    def test_init_folder(self):
        class_ids = import_data_from_roboflow.init_from_folder("test_vectors/sample_annotations_folder/")
        assert len(class_ids) == 3
    
    def test_init_roboflow(self):
        class_ids = import_data_from_roboflow.init_from_roboflow(self.workspace_name, self.project_name, self.api_key)
        assert len(class_ids) == 3
    
    def test_list_images_masks(self):
        print(import_data_from_roboflow.COCO_PATH)
        assert len(import_data_from_roboflow.list_all_labels()) == 3

    def test_preprocess_imgs(self):
        import_data_from_roboflow.preprocess_images(self.original_images_path, self.image_dataset_folder_path)
        fpath = import_data_from_roboflow.get_image_dataset_folder_path()
        assert fpath == self.image_dataset_folder_path

    def test_list_all_images(self):
        self.set_up_global_vars()
        image_names = import_data_from_roboflow.list_all_images()
        assert len(image_names) == 6
    
    def test_get_image(self):
        import_data_from_roboflow.get_image("y0124_png.rf.dbeef95400a14da233736fc6e7df31b9.jpg")

    def test_get_mask(self):
        m = import_data_from_roboflow.get_mask("y0127_png.rf.d5e3bec4667f62e119206f3a34f617e0.jpg", 1)
        assert np.any(np.array(m))

    def test_mask_volume(self):
        segs = import_data_from_roboflow.create_mask_volume(1)
        assert np.any(segs[0])

    # Model functions
    
    def set_up_model(self):
        self.setUp()
        self.model = propagate_mask_medsam2.CustomMEDSAM2(self.MODEL_CONFIG, self.MODEL_CHECKPOINT)

    def test_forward_pass(self):
        self.set_up_model()
        self.set_up_global_vars()
        output_masks = self.model._propagate_single_direction(self.image_dataset_folder_path, self.seg_vol_1)
        for f in output_masks:
            assert np.any(f)

    def test_backward_pass(self):
        self.set_up_model()
        self.set_up_global_vars()
        output_masks = self.model._propagate_single_direction(self.image_dataset_folder_path, self.seg_vol_1, reverse=True)
        for f in output_masks:
            assert np.any(f)

    def test_get_kf_from_seg(self):
        segmentations = np.array([
            np.full((3, 4), np.nan),   
            np.ones((3, 4)),
            np.ones((3, 4)), 
            np.full((3, 4), np.nan)   
        ])
        kf_idx = self.model._get_keyframe_indices_from_sparse_segmentations(segmentations)
        assert kf_idx[0] == 1
        assert kf_idx[1] == 2
    
    def test_propagate(self):
        self.set_up_model()
        self.set_up_global_vars()
        output_masks = self.model.propagate(self.image_dataset_folder_path, self.seg_vol_1)
        for f in output_masks:
            assert np.any(f)

    def test_propagate_with_smoothing(self):
        self.set_up_model()
        self.set_up_global_vars()
        output_masks = self.model.propagate(self.image_dataset_folder_path, self.seg_vol_1, 1, 3, 5)
        for f in output_masks:
            assert np.any(f)
    
    def test_combine_class_masks(self):
        self.set_up_model()
        self.set_up_global_vars()
        output_masks0 = self.model.propagate(self.image_dataset_folder_path, self.seg_vol_0)
        output_masks1 = self.model.propagate(self.image_dataset_folder_path, self.seg_vol_1)
        output_dir = "test_vectors/output_combined_masks/"
        propagate_mask_medsam2.combine_class_masks([output_masks0, output_masks1], output_dir=output_dir, show=True)

    def test_output_coco_tiff(self):
        self.set_up_model()
        self.set_up_global_vars()
        fused_masks1 = self.model.propagate(self.image_dataset_folder_path, self.seg_vol_1)
        export_coco.save_segmentations_as_coco([fused_masks1], "test_vectors/predicted_segmentations_coco.json")
        export_coco.coco_to_tiff("test_vectors/predicted_segmentations_coco.json", "test_vectors/output_volume.tiff")

    # Analyze volumes
    def test_dist_heatmap_2d(self):
        toy_mask = np.zeros((5,10,12))
        toy_mask[0,0,0] = 1
        map_2d = analyze_volumes.generate_distance_heatmap(toy_mask, distance_threshold_px_near=2, distance_threshold_px_far=5, 
                              near_color_rgb=(163, 222, 153), far_color_rgb=(205, 164, 224),
                              overlay=True, show=False, 
                              output_path=None, use_2d_xy_distances=True)
        assert map_2d[0,0,0,0]==255
        assert map_2d[1,0,0,0]==0

    def test_dist_heatmap_3d(self):
        toy_mask = np.zeros((5,10,12))
        toy_mask[0,0,0] = 1
        map_3d = analyze_volumes.generate_distance_heatmap(toy_mask, distance_threshold_px_near=2, distance_threshold_px_far=5, 
                              near_color_rgb=(163, 222, 153), far_color_rgb=(205, 164, 224),
                              overlay=True, show=False, 
                              output_path=None, use_2d_xy_distances=False)
        assert map_3d[0,0,0,0]== 255
        assert map_3d[1,0,0,0]==163

    def test_close_map_2d_3d(self):
        toy_mask1 = np.zeros((5,10,12))
        toy_mask1[0,0,0] = 1
        toy_mask2 = np.zeros((5,10,12))
        toy_mask2[5,0,0] = 1
        map_3d = analyze_volumes.regions_close_to_object_types([toy_mask1, toy_mask2],
                                                                thresh=5, use_2d_xy_distances=False)
        assert map_3d[3,0,0] == 1
