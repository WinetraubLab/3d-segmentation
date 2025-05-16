import numpy as np
import numpy.testing as npt
import unittest
import import_data_from_roboflow
import propagate_mask_medsam2
import cv2

class TestFitPlaneElastic(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        
    def test_main_function_runs(self):
        class_ids = import_data_from_roboflow.init_from_folder("test_vectors/sample_annotations_folder/")
        assert len(class_ids) == 3
    
    def test_list_images_masks(self):
        print(import_data_from_roboflow.COCO_PATH)
        assert len(import_data_from_roboflow.list_all_labels()) == 3

    def test_preprocess_imgs(self):
        import_data_from_roboflow.preprocess_images("test_vectors/sample_image_sequence/", "test/vectors/preprocessed_images/")