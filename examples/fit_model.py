
from protodata import datasets
from protodata.utils import get_data_location
import sys
import shutil
import logging
import os

from layout import kernel_example_layout_fn, example_layout_fn

from training.fit_validate import DeepNetworkValidation
from training.fit import DeepNetworkTraining
from training.policy import CyclicPolicy, InverseCyclingPolicy

from validation.tuning import _run_setting

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
)
logging.getLogger().addHandler(logging.FileHandler('file.log'))

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    mode = int(sys.argv[1])
    # folder = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb01/walle/aus'
    folder = 'incremental_4l'

    params = {
        'l2_ratio': 0.00342859,
        'lr': 0.000932367,
        'lr_decay': 0.322196,
        'lr_decay_epochs': 500,
        'memory_factor': 2,
        'hidden_units': 2048,
        'n_threads': 4,
        'kernel_size': 512,
        'kernel_mean': 0.0,
        'kernel_std': 0.81609644,
        'strip_length': 5,
        'batch_size': 128,
        'num_layers': 3,
        'train_epochs': [525, 790, 2185],
        'network_fn': kernel_example_layout_fn
    }

    settings = datasets.MotorSettings
    dataset = datasets.Datasets.MOTOR

    if mode == 0:

        logger.info('Running training ...')

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        m = DeepNetworkTraining(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )

        m.fit(
            **params
        )

    elif mode == 1:

        logger.info('Running training with early stop on validation ...')

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        m = DeepNetworkValidation(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )
        m.fit(train_folds=range(9), val_folds=[9], layerwise=False, **params)

    elif mode == 2:
        
        logger.info('Running prediction ...')

        m = DeepNetworkValidation(
            folder=folder,
            settings_fn=settings,
            data_location=get_data_location(dataset, folded=True)
        )

        res = m.predict(
            **params
        )

        logger.info('Got results {} for prediction'.format(res))

    elif mode == 3:
        
        logger.info('Running incremental training...')

        _run_setting(
                dataset,
                settings,
                params,
                folder='incremental_motor_4l',
                n_runs=1,
                test_batch_size=128,
                fine_tune=None
        )

    else:

        logger.error(
            'Invalid option %d. Modes are:\n' % mode +
            '\t - 0 for traditional training' +
            '\t - 1 for training with validation' +
            '\t - 2 for prediction'
        )
