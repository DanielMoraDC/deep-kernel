from hyperopt import hp
import numpy as np
import logging

from protodata import datasets

from validation.tuning import tune_model
from validation.fine_tuning import FineTuningType
from training.policy import CyclicPolicy
from layout import kernel_example_layout_fn

CV_TRIALS = 25
SIM_RUNS = 10
MAX_EPOCHS = 10000

n_layers = 2

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='motor_%dl' % n_layers,
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
)

if __name__ == '__main__':

    search_space = {
        'batch_size': 2 ** hp.choice('batch_size_log2', [7]),
        'l2_ratio': 10 ** hp.uniform('l2_log10', -4, -2),
        'lr': 10 ** hp.uniform('lr_log10', -5, -3),
        'kernel_size': 2 ** (9 + hp.randint('kernel_size_log2', 3)),
        'kernel_std': hp.uniform('kernel_std_log10', 1e-2, 1.0),
        'hidden_units': 2 ** (9 + hp.randint('hidden_units_log2', 3)),
        'lr_decay': hp.uniform('lr_decay', 0.1, 1.0),
        'lr_decay_epochs': hp.uniform('lr_decay_epochs', 100, 1000),
    }

    # Fixed parameters
    search_space.update({
        'max_layers': n_layers,
        'n_threads': 4,
        'memory_factor': 2,
        'batch_norm': False,
        'max_epochs': MAX_EPOCHS,
        'strip_length': 5,
        'progress_thresh': 0.1,
        'kernel_mean': 0.0,
        'network_fn': kernel_example_layout_fn
    })

    stats = tune_model(
        dataset=datasets.Datasets.MOTOR,
        settings_fn=datasets.MotorSettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=False,
        folder='motor',
        runs=SIM_RUNS,
        test_batch_size=1,
        #fine_tune=FineTuningType.ExtraLayerwise(
        #    epochs_per_layer=20, policy=CyclicPolicy
        #)
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        logger.info('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
