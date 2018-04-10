
from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from training.policy import CyclicPolicy
from validation.tuning import tune_model
from layout import cnn_kernel_example_layout_fn

CV_TRIALS = 10
SIM_RUNS = 10
MAX_EPOCHS = 10000

n_layers = 2
name='cifar10_%dl' % n_layers

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=name,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(message)s',
)

if __name__ == '__main__':

    search_space = {
        # CNN params
        'cnn_filter_size': hp.choice('cnn_filter_size', [3, 5, 7]),
        'map_size': 2 ** (5 + hp.randint('map_size', 3)),
        'stride': hp.choice('stride', [4]),
        # CNN kernel params
        'cnn_kernel_size': 2 ** (6 + hp.randint('cnn_kernel_size_log2', 3)),
        # Shared kernel params
        'kernel_size': 2 ** (9 + hp.randint('kernel_size_log2', 3)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (8 + hp.randint('hidden_units_log2', 3)),
        # Training params
        'batch_size': 2 ** (6 + hp.randint('batch_size_log2', 3)),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -5, -3),
        'lr': 10 ** hp.uniform('lr_log10', -5, -3),
        'lr_decay': hp.uniform('lr_decay', 0.1, 1.0),
        'lr_decay_epochs': hp.uniform('lr_decay_epochs', 25, 70),
        'epochs_per_layer': 10 + hp.randint('epochs_per_layer', 20)
    }

    # Fixed parameters
    search_space.update({
        'image_specs': {
            'scale_size': 32,
            'crop_size': 32,
            'random_crop': False,
            'channels': 3,
            'isotropic': False,
            'mean': [0.0, 0.0, 0.0] # Already normalizing inputs during fit
        },
        'cnn_batch_norm': False,
        'batch_norm': False,
        'padding': 'VALID',
        'num_layers': n_layers,  # These are cnn layers here
        'layerwise_progress_thresh': 0.1,
        'n_threads': 4,
        'memory_factor': 2,
        'kernel_dropout_rate': None,
        'max_epochs': MAX_EPOCHS,
        'strip_length': 5,
        'kernel_dropout_rate': 0.95,
        'tune_folder': name + '_validate',
        'progress_thresh': 0.1,
        'network_fn': cnn_kernel_example_layout_fn,
         'policy': {'switch_policy': CyclicPolicy},
        'fc_layers': 2
    })

    stats = tune_model(
        dataset=datasets.Datasets.CIFAR10,
        settings_fn=datasets.Cifar10Settings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=False,
        folder=name + '_stats',
        runs=SIM_RUNS,
        test_batch_size=128
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))

