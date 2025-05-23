"""
Unit tests for data_processing module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_processing import create_windows, sliding_window, normalize_windows

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing"""
        # Create a simple dataset with known values and labels
        self.sample_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'class': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  # Binary labels
        })
        
        # Expected normalized data (manually calculated)
        self.expected_normalized = np.array([
            [0.0, 0.0],
            [0.1111111, 0.1111111],
            [0.2222222, 0.2222222],
            [0.3333333, 0.3333333],
            [0.4444444, 0.4444444],
            [0.5555556, 0.5555556],
            [0.6666667, 0.6666667],
            [0.7777778, 0.7777778],
            [0.8888889, 0.8888889],
            [1.0, 1.0]
        ])
        
        # Expected labels
        self.expected_labels_step1 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        self.expected_labels_step2 = np.array([0, 1, 0, 1])
    
    def test_create_windows_with_scaler(self):
        """Test create_windows with MinMaxScaler"""
        # Test with step_size=1
        windows, labels, scaler = create_windows(self.sample_data, window_length=3, step_size=1)
        
        # Check shapes
        self.assertEqual(windows.shape, (8, 3, 2))
        self.assertEqual(labels.shape, (8,))
        
        # Check labels
        np.testing.assert_array_equal(labels, self.expected_labels_step1)
        
        # Check normalization (values should be between 0 and 1)
        self.assertTrue(np.all(windows >= 0))
        self.assertTrue(np.all(windows <= 1))
        
        # Check scaler
        self.assertIsInstance(scaler, MinMaxScaler)
        
        # Test with step_size=2
        windows, labels, scaler = create_windows(self.sample_data, window_length=3, step_size=2)
        
        # Check shapes
        self.assertEqual(windows.shape, (4, 3, 2))
        self.assertEqual(labels.shape, (4,))
        
        # Check labels
        np.testing.assert_array_equal(labels, self.expected_labels_step2)
    
    def test_create_windows_with_provided_scaler(self):
        """Test create_windows with a provided scaler"""
        # Create and fit a scaler
        scaler = MinMaxScaler()
        scaler.fit(self.sample_data[['A', 'B']])
        
        # Use the pre-fitted scaler
        windows, labels, returned_scaler = create_windows(
            self.sample_data, window_length=3, step_size=1, scaler=scaler
        )
        
        # Check that the returned scaler is the same object
        self.assertIs(scaler, returned_scaler)
        
        # Check normalization
        self.assertTrue(np.all(windows >= 0))
        self.assertTrue(np.all(windows <= 1))
    
    def test_sliding_window_with_scaler(self):
        """Test sliding_window function with MinMaxScaler"""
        # Test with stride=1
        windows, labels, scaler = sliding_window(self.sample_data, window_length=3, stride=1)
        
        # Check shapes
        self.assertEqual(windows.shape, (8, 3, 2))
        self.assertEqual(labels.shape, (8,))
        
        # Check labels
        np.testing.assert_array_equal(labels, self.expected_labels_step1)
        
        # Check normalization
        self.assertTrue(np.all(windows >= 0))
        self.assertTrue(np.all(windows <= 1))
        
        # Check scaler
        self.assertIsInstance(scaler, MinMaxScaler)
    
    def test_inverse_transform(self):
        """Test inverse transform functionality"""
        # Create windows with scaler
        windows, _, scaler = create_windows(self.sample_data, window_length=3, step_size=1)
        
        # Get first window
        first_window = windows[0]
        
        # Inverse transform
        original_data = scaler.inverse_transform(first_window)
        
        # Check shape
        self.assertEqual(original_data.shape, (3, 2))
        
        # Check values (should be close to original data)
        expected_original = self.sample_data[['A', 'B']].values[:3]
        np.testing.assert_allclose(original_data, expected_original, rtol=1e-5)
    
    def test_error_handling_with_scaler(self):
        """Test error handling for scaler-related issues"""
        # Test missing label column
        data_no_label = self.sample_data.drop(columns=['class'])
        with self.assertRaises(ValueError):
            create_windows(data_no_label, window_length=3)
        
        # Test sliding_window with numpy array (should raise error)
        with self.assertRaises(ValueError):
            sliding_window(self.sample_data.values, window_length=3)
    
    def test_normalize_windows(self):
        """Test normalize_windows (unchanged from original)"""
        windows, _, _ = create_windows(self.sample_data, window_length=3, step_size=2)
        normalized = normalize_windows(windows, method='minmax')
        
        # Check shape
        self.assertEqual(normalized.shape, windows.shape)
        
        # Check min-max scaling (should be between 0 and 1)
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[2]):
                self.assertAlmostEqual(np.min(normalized[i, :, j]), 0.0)
                self.assertAlmostEqual(np.max(normalized[i, :, j]), 1.0)

if __name__ == '__main__':
    unittest.main()