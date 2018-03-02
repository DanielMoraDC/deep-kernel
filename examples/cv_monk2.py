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

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='monk2_alternate_%d.log' % n_layers,
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
)


if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** hp.choice('batch_size_log2', [4]),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -4, -2),
        'lr': 10 ** hp.uniform('l2_log10', -4, -2),
        'kernel_size': 2 ** (5 + hp.randint('kernel_size_log2', 3)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (6 + hp.randint('hidden_units_log2', 3)),
        'epochs_per_layer': 25 + hp.randint('epochs_per_layer', 25),
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
        'strip_length': 5,
        'memory_factor': 1,
        'batch_norm': False,
        'max_epochs': MAX_EPOCHS,
        'progress_thresh': 0.1,
        'kernel_mean': 0.0,
    })

    stats = tune_model(
        dataset=datasets.Datasets.MONK2,
        settings_fn=datasets.Monk2Settings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=True,
        folder='monk',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
