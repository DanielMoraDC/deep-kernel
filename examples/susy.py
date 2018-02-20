from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from validation.tuning import tune_model
from training.policy import CyclicPolicy

CV_TRIALS = 5
SIM_RUNS = 10
MAX_EPOCHS = 10000

n_layers = 1

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='susy_alternate_%d.log' % n_layers,
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
)

if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** (7 + hp.uniform('batch_size_log2', 2)),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -5, -2),
        'lr': 10 ** hp.uniform('l2_log10', -5, -3),
        'kernel_size': 2 ** (8 + hp.randint('kernel_size_log2', 3)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (9 + hp.randint('hidden_units_log2', 3)),
        'epochs_per_layer': 10 + hp.randint('epochs_per_layer', 75)
    }

    # Fixed parameters
    search_space.update({
        'num_layers': n_layers,
        'lr_decay': 0.5,
        'lr_decay_epocs': 250,
        'n_threads': 4,
        'memory_factor': 2,
        'max_epochs': MAX_EPOCHS,
        'strip_length': 5,
        'progress_thresh': 0.1,
        'kernel_mean': 0.0,
        'switch_policy': CyclicPolicy,
        'policy_seed': np.random.randint(1, 1000)
    })

    stats = tune_model(
        dataset=datasets.Datasets.SUSY,
        settings_fn=datasets.SusySettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=False,
        folder='susy',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
