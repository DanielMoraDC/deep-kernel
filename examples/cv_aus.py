from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from validation.tuning import tune_model
from training.policy import CyclicPolicy, InverseCyclingPolicy, RandomPolicy
from layout import kernel_example_layout_fn

CV_TRIALS = 25
SIM_RUNS = 10
MAX_EPOCHS = 10000

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='aus_baseline.log',
    level=logging.INFO
)


if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** (4 + hp.randint('batch_size_log2', 2)),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -4, -2),
        'lr': 10 ** hp.uniform('lr_log10', -5, -3),
        'kernel_size': 2 ** (5 + hp.randint('kernel_size_log2', 2)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (5 + hp.randint('hidden_units_log2', 2)),
        'lr_decay': hp.uniform('lr_decay', 0.1, 1.0),
        'lr_decay_epochs': hp.uniform('lr_decay_epochs', 100, 1000),
        # Comment next lines for non-layerwise training
        '''policy': hp.choice('policy', [
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
        ])'''
    }

    # Fixed parameters
    search_space.update({
        'num_layers': 1,
        'layerwise_progress_thresh': 0.1,  # Only used if layerwise
        'n_threads': 4,
        'strip_length': 5,
        'memory_factor': 1,
        'max_epochs': MAX_EPOCHS,
        'progress_thresh': 0.1,
        'network_fn': kernel_example_layout_fn
    })

    stats = tune_model(
        dataset=datasets.Datasets.AUS,
        settings_fn=datasets.AusSettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=True,
        layerwise=False,
        folder='aus',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
