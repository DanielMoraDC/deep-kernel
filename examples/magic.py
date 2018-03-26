from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from validation.tuning import tune_model
from training.policy import CyclicPolicy, InverseCyclingPolicy, RandomPolicy

CV_TRIALS = 25
SIM_RUNS = 10
MAX_EPOCHS = 10000

n_layers = 1

logging.basicConfig(
        level=logging.INFO,
        filename='magic_alternate_%d.log' % n_layers
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** hp.choice('batch_size_log2', [7]),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -4, 0),
        'lr': 10 ** hp.uniform('lr_log10', -4, -2),
        'kernel_size': 2 ** (6 + hp.randint('kernel_size_log2', 4)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (9 + hp.randint('hidden_units_log2', 3)),
        'epochs_per_layer': 10 + hp.randint('epochs_per_layer', 75),
        'lr_decay': hp.uniform('lr_decay', 0.1, 1.0),
        'lr_decay_epochs': hp.uniform('lr_decay_epochs', 100, 1000),
        # Comment next lines for non-layerwise training
        'policy': hp.choice('policy', [
            {
                'switch_policy': CyclicPolicy
            },
            {
                'switch_policy': InverseCyclingPolicy
            },
            {
                'switch_policy': RandomPolicy,
                'policy_seed': hp.randint('seed', 10000)
            }
        ])
    }

    # Fixed parameters
    search_space.update({
        'num_layers': n_layers,
        'n_threads': 4,
        'batch_norm': False,
        'memory_factor': 2,
        'max_epochs': MAX_EPOCHS,
        'strip_length': 5,
        'progress_thresh': 0.1
    })

    stats = tune_model(
        dataset=datasets.Datasets.MAGIC,
        settings_fn=datasets.MagicSettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=False,
        folder='magic',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
