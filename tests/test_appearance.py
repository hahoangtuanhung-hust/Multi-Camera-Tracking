import unittest

import numpy as np

from multicam_tracker.appearance import (
    appearance_similarity,
    extract_person_descriptor,
    update_appearance_prototype,
)


class AppearanceDescriptorTests(unittest.TestCase):
    def test_same_colour_is_more_similar_than_different_colour(self):
        red = np.zeros((80, 40, 3), dtype=np.uint8)
        blue = np.zeros_like(red)
        red[:, :] = (0, 0, 255)
        blue[:, :] = (255, 0, 0)
        box = np.array([0.0, 0.0, 40.0, 80.0])

        first = extract_person_descriptor(red, box)
        second = extract_person_descriptor(red.copy(), box)
        different = extract_person_descriptor(blue, box)

        self.assertGreater(appearance_similarity(first, second), 0.99)
        self.assertLess(appearance_similarity(first, different), 0.25)

    def test_prototype_update_stays_normalized(self):
        first = np.array([1.0, 0.0, 0.0])
        second = np.array([0.8, 0.2, 0.0])
        updated = update_appearance_prototype(first, second, alpha=0.25)
        self.assertAlmostEqual(float(np.linalg.norm(updated)), 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
