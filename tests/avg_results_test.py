import unittest
import numpy as np

from validation.tuning import _average_results

TOLERANCE = 0.05  # 5% tolerance

LAYERWISE_DATA = [
    {
        'train_loss': 0.20,
        'val_loss': 0.40,
        'train_error': 0.05,
        'val_error': 0.10,
        'train_epochs': [25, 60, 120, 200]
    },
    {
        'train_loss': 0.50,
        'val_loss': 0.600,
        'train_error': 0.10,
        'val_error': 0.15,
        'train_epochs': [100, 120, 190]
    },
    {
        'train_loss': 0.20,
        'val_loss': 0.40,
        'train_error': 0.05,
        'val_error': 0.10,
        'train_epochs': [50, 70]
    }
]

LAYERWISE_GROUNDTRUTH = {
    'train_loss': 0.2999,
    'val_loss': 0.4666,
    'train_error': 0.0666,
    'val_error': 0.1166,
    'train_epochs': [50.0, 70.0, 155.0, 200.0]
}

BASE_DATA = [
    {
        'train_loss': 0.20,
        'val_loss': 0.40,
        'train_error': 0.05,
        'val_error': 0.10,
        'epoch': 70
    },
    {
        'train_loss': 0.50,
        'val_loss': 0.60,
        'train_error': 0.10,
        'val_error': 0.15,
        'epoch': 120
    },
    {
        'train_loss': 0.20,
        'val_loss': 0.40,
        'train_error': 0.05,
        'val_error': 0.10,
        'epoch': 50
    }
]

BASE_GROUNDTRUTH = {
    'train_loss': 0.2999,
    'val_loss': 0.4666,
    'train_error': 0.0666,
    'val_error': 0.1166,
    'epoch': 70.0,
}


class AvgResultsTestCase(unittest.TestCase):

    def assert_equal_dicts(self, dict1, dict2):
        if set(dict1.keys()) != set(dict2.keys()):
            return False

        for k, v in dict1.items():
            try:
                if not np.isclose(dict2[k], v, rtol=TOLERANCE):
                    return False
            except ValueError:
                if not np.array_equal(dict2[k], v):
                    return False

        return True

    def test_avg_base(self):
        self.assertTrue(
            self.assert_equal_dicts(
                _average_results(BASE_DATA), BASE_GROUNDTRUTH
            )
        )

    def test_avg_layerwise(self):
        self.assertTrue(
            self.assert_equal_dicts(
                _average_results(LAYERWISE_DATA), LAYERWISE_GROUNDTRUTH
            )
        )


if __name__ == '__main__':
    unittest.main()
