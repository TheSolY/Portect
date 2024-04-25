import unittest
import utils
import numpy as np


mat_a = np.array([[0.3, 0.4, 0.9, 0.2, 0.44],
                  [0.1, 0.5, 1.0, 0.6, 0.7],
                  [0.2, 0.5, 1.0, 0.6, 0.7],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0]])

vec_a = np.array([0.1, 0.5, 1.0, 0.6, 0.7])


class RowCosineSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.vec_a = vec_a
        self.mat_a = mat_a

    def test_same_as_row(self):
        res = utils.row_cosine_similarity(self.vec_a, self.mat_a)
        self.assertAlmostEqual(res[1], 1.0)


class FarthestNeighborTest(unittest.TestCase):
    def setUp(self):
        self.vec_a = vec_a
        self.mat_a = mat_a

    def test_same_as_row(self):
        res_idx, res_vec = utils.farthest_neighbor(self.vec_a, self.mat_a)
        self.assertEqual(res_idx, 1)
        self.assertTrue(np.array_equal(res_vec, self.vec_a))


if __name__ == '__main__':
    unittest.main()

