from hyperopt import hp
import numpy as np

from protodata import datasets
from validation.layerwise import tune_model

CV_TRIALS = 25
SIM_RUNS = 10
MAX_EPOCHS = 10000


if __name__ == '__main__':

    search_space = {
        'batch_size': hp.choice('batch_size', [16, 32]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [1e-2, 1e-3, 1e-4]),
        'lr': hp.choice('lr', [1e-3, 1e-4, 1e-5]),
        'kernel_size': hp.choice('kernel_size', [32, 64]),
        'kernel_std': hp.choice('kernel_std', [1e-2, 0.1, 0.25, 0.5, 1.0]),
        'hidden_units': hp.choice('hidden_units', [32, 64])
    }

    # Fixed parameters
    search_space.update({
        'max_layers': 5,
        'lr_decay': 0.5,
        'lr_decay_epocs': 250,
        'n_threads': 4,
        'strip_length': 5,
        'memory_factor': 1,
        'max_epochs': MAX_EPOCHS,
        'kernel_mean': 0.0
    })

    stats = tune_model(
        dataset=datasets.Datasets.BALANCE,
        settings_fn=datasets.BalanceSettings,
        search_space=search_space,
        n_trials=CV_TRIALS,
        cross_validate=True,
        folder='balance',
        runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        print('%s: %f +- %f' % (m, np.mean(values), np.std(values)))