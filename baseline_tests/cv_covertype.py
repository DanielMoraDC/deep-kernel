from hyperopt import hp
import numpy as np

from protodata import datasets
from model_validation import evaluate_model

CV_TRIALS = 5
SIM_RUNS = 10
MAX_EPOCHS = 10000


if __name__ == '__main__':

    search_space = {
        'batch_size': hp.choice('batch_size', [128, 256]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [1e-3, 1e-4, 1e-5]),
        'lr': hp.choice('lr', [1e-3, 1e-4, 1e-5]),
        # LR bigger or equal than 1e-3 seem to be too high
        'kernel_size': hp.choice('kernel_size', [256, 512, 1024]),
        'kernel_std': hp.choice('kernel_std', [1e-2, 0.1, 0.5, 1.0]),
        'hidden_units': hp.choice('hidden_units', [512, 1024, 2048])
    }

    # Fixed parameters
    search_space.update({
        'n_threads': 4,
        'memory_factor': 2,
        'strip_length': 5,
        'progress_thresh': 0.1
    })

    all_stats = evaluate_model(
        dataset=datasets.Datasets.COVERTYPE,
        settings=datasets.CoverTypeSettings,
        search_space=search_space,
        max_epochs=MAX_EPOCHS,
        output_folder=None,
        cv_trials=CV_TRIALS,
        runs=SIM_RUNS
    )

    metrics = all_stats[0].keys()
    for m in metrics:
        values = [x[m] for x in all_stats]
        print('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
