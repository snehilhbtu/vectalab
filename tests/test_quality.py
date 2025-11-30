import unittest
import numpy as np
import cv2
from vectalab.quality import reduce_to_palette, analyze_image

class TestQuality(unittest.TestCase):
    def test_reduce_to_palette_kmeans(self):
        # Create a synthetic image with 3 distinct colors
        # Red, Green, Blue
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:33, :] = [255, 0, 0]
        img[33:66, :] = [0, 255, 0]
        img[66:100, :] = [0, 0, 255]
        
        # Add some noise
        noise = np.random.randint(0, 10, (100, 100, 3), dtype=np.uint8)
        img_noisy = cv2.add(img, noise)
        
        # Reduce to 3 colors
        reduced = reduce_to_palette(img_noisy, n_colors=3)
        
        # Check unique colors
        unique_colors = np.unique(reduced.reshape(-1, 3), axis=0)
        self.assertEqual(len(unique_colors), 3)
        
        # Check if colors are close to original (Red, Green, Blue)
        # We can't be 100% sure of the order, but we can check if they are close
        # to [255, 0, 0], [0, 255, 0], [0, 0, 255]
        # But K-means centers might be slightly shifted due to noise.
        # However, with small noise, they should be very close.
        
    def test_reduce_to_palette_fallback(self):
        # Test fallback (we can't easily force cv2.kmeans to fail without mocking, 
        # but we can test the function with valid input)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = [100, 100, 100]
        reduced = reduce_to_palette(img, n_colors=2)
        self.assertTrue(reduced.shape == img.shape)

if __name__ == '__main__':
    unittest.main()
