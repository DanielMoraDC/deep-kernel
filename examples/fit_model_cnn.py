
from protodata import datasets
from protodata.utils import get_data_location

import sys
import shutil
import logging
import os

from layout import cnn_kernel_example_layout_fn, cnn_example_layout_fn

from training.fit_validate import DeepNetworkValidation
from training.fit import DeepNetworkTraining
from training.policy import CyclicPolicy

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
)
logging.getLogger().addHandler(logging.FileHandler('file.log'))

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


if __name__ == '__main__':

    mode = int(sys.argv[1])
    # folder = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb01/walle/aus'
    folder = 'test_aux'

    params = {
        'image_specs': {
            'batch_size': BATCH_SIZE,
            'crop_size': 28,
            'scale_size': 28,
            'isotropic': False,
            'mean': [0.0, 0.0, 0.0],
            'random_crop': False
        },
        'l2_ratio': 1e-3,
        'lr': 1e-5,
        'lr_decay': 0.5,
        'lr_decay_epochs': 50,
        'memory_factor': 2,
        'hidden_units': 512,
        'n_threads': 4,
        'map_size': 256,
        'cnn_filter_size': 5,
        'cnn_kernel_size': 256,
        'kernel_size': 512,
        'kernel_std': 0.1,
        'strip_length': 2,
        'cnn_batch_norm': False,
        'batch_norm': True,
        'padding': 'VALID',
        'kernel_dropout_rate': None,
        'batch_size': BATCH_SIZE,
        'num_layers': 2,
        'epochs_per_layer': 5,
        'max_epochs': 250,
        'switch_policy': CyclicPolicy,
        'max_successive_strips': 3,
        'network_fn': cnn_kernel_example_layout_fn,
    }

    settings = datasets.FashionMnistSettings
    dataset = datasets.Datasets.FASHION_MNIST

    if mode == 0:

        logger.info('Running training ...')

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        m = DeepNetworkTraining(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )

        m.fit(**params)

    elif mode == 1:

        logger.info('Running training with early stop on validation ...')

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        m = DeepNetworkValidation(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )
        m.fit(train_folds=range(9), val_folds=[9], layerwise=True, **params)

    elif mode == 2:

        m = DeepNetworkValidation(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )

        res = m.predict(
            **params
        )

        logger.info('Got results {} for prediction'.format(res))

    else:

        logger.error(
            'Invalid option %d. Modes are:\n' % mode +
            '\t - 0 for traditional training' +
            '\t - 1 for training with validation' +
            '\t - 2 for prediction'
        )
