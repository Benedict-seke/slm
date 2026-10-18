import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eval_mvtec_ad import (
    _classifier_tensor,
    _embedding_knn_scores,
    _nearest_neighbor_map,
    _roc_auc,
    _texture_features,
    discover_categories,
)


class MvtecEvaluationTests(unittest.TestCase):
    def test_roc_auc_handles_order_and_ties(self):
        self.assertEqual(_roc_auc(np.array([0.1, 0.9]), np.array([False, True])), 1.0)
        self.assertEqual(_roc_auc(np.array([0.5, 0.5]), np.array([False, True])), 0.5)

    def test_discovers_categories_and_ignores_other_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            category = root / "bottle"
            (category / "train" / "good").mkdir(parents=True)
            (category / "test" / "good").mkdir(parents=True)
            (category / "ground_truth").mkdir()
            (root / "unrelated").mkdir()
            self.assertEqual(discover_categories(root), [category])

    def test_detector_maps_have_expected_shapes(self):
        image = np.zeros((16, 16, 3), dtype=np.float32)
        texture = _texture_features(image)
        nearest = _nearest_neighbor_map(image, image[None, ...])
        self.assertEqual(texture.shape, (16, 16, 3))
        self.assertTrue(np.isfinite(texture).all())
        self.assertEqual(nearest.shape, (16, 16))
        self.assertEqual(float(nearest.max()), 0.0)

    def test_classifier_preprocessing_matches_model_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.new("RGB", (300, 200), (100, 120, 140)).save(image_path)
            tensor = _classifier_tensor(image_path)
            self.assertEqual(tensor.shape, (1, 3, 224, 224))
            self.assertTrue(np.isfinite(tensor).all())

    def test_embedding_knn_scores_nearest_reference_lowest(self):
        references = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        queries = np.array([[0.0, 0.0], [20.0, 20.0]], dtype=np.float32)
        scores = _embedding_knn_scores(references, queries)
        self.assertEqual(scores.shape, (2,))
        self.assertLess(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
