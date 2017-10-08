from hyperopt import hp
from protodata.datasets import Datasets, TitanicSettings

from cross_validation import cross_validate

CV_TRIALS = 25
N_FOLDS = 10


def perform_cv(dataset, settings):
    # Variable parameters
    search_space = {
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [0, 1e-1, 1e-2, 1e-3]),
        'lr': hp.choice('lr', [1e-1, 1e-2, 1e-3]),
        'hidden_units': hp.choice('hidden_units', [64, 128, 256])
    }

    # Fixed parameters
    search_space.update({
        'n_threads': 2,
        'memory_factor': 2,
        'validation_interval': 250,
        'max_steps': 50000,
        'train_tolerance': 1e-3
    })

    trials, best, params = cross_validate(dataset=dataset,
                                          settings=settings,
                                          n_folds=N_FOLDS,
                                          n_trials=CV_TRIALS,
                                          search_space=search_space)

    stats = trials.best_trial['result']['averaged']
    return stats, params


if __name__ == '__main__':
    perform_cv(Datasets.TITANIC, TitanicSettings)
