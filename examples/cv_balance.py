from hyperopt import hp
import numpy as np

from protodata import datasets
from cross_validation import evaluate_model

CV_TRIALS = 10
SIM_RUNS = 10


if __name__ == '__main__':

    search_space = {
        'batch_size': hp.choice('batch_size', [16]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [0, 1e-1, 1e-2, 1e-3]),
        'lr': hp.choice('lr', [1e-1, 1e-2, 1e-3]),
        'hidden_units': hp.choice('hidden_units', [128, 256])
    }

    # Fixed parameters
    search_space.update({
        'n_threads': 4,
        'memory_factor': 1,
        'strip_length': 5,
        'max_epochs': 1000,
        'progress_thresh': 0.1
    })

    print('Before evaluation')

    all_stats = evaluate_model(
        datasets.Datasets.BALANCE,
        datasets.BalanceSettings,
        search_space,
        '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/ev',
        cv_trials=CV_TRIALS,
        runs=SIM_RUNS
    )

    metrics = all_stats[0].keys()
    for m in metrics:
        values = [x[m] for x in all_stats]
        print('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
