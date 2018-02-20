from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from validation.tuning import tune_model
from training.policy import CyclicPolicy

CV_TRIALS = 25
SIM_RUNS = 10
MAX_EPOCHS = 10000

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='titanic_baseline.log',
    level=logging.INFO
)


if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** (4 + hp.randint('batch_size_log2', 2)),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -4, -1),
        'lr': 10 ** hp.uniform('l2_log10', -4, -2),
        'kernel_size': 2 ** (5 + hp.randint('kernel_size_log2', 3)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (6 + hp.randint('hidden_units_log2', 3))
    }

    # Fixed parameters
    search_space.update({
        'num_layers': 1,
        'layerwise_progress_thresh': 0.1,
        'lr_decay': 0.5,
        'lr_decay_epochs': 250,
        'n_threads': 4,
        'strip_length': 5,
        'memory_factor': 1,
        'max_epochs': MAX_EPOCHS,
        'progress_thresh': 0.1,
        'kernel_mean': 0.0,
        'switch_policy': CyclicPolicy,
        'policy_seed': np.random.randint(1, 1000)
    })

    stats = tune_model(
        dataset=datasets.Datasets.TITANIC,
        settings_fn=datasets.TitanicSettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=True,
        layerwise=False,
        folder='titanic',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
