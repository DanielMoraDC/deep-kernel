from hyperopt import hp
from protodata.datasets import Datasets, TitanicSettings

from cross_validation import cross_validate

CV_ROOT_FOLDER = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/cv'  # noqa
CV_TRIALS = 100
N_FOLDS = 10


def perform_cv(dataset, settings):
    # Variable parameters
    search_space = {
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [0, 1e-1, 1e-2, 1e-3]),
        'lr': hp.choice('lr', [1e-1, 1e-2, 1e-3, 1e-4]),
        'hidden_units': hp.choice('hidden_units', [64, 128, 256])
    }

    # Fixed parameters
    search_space.update({
        'n_threads': 2,
        'memory_factor': 2,
        'summary_steps': 100,
        'validation_interval': 250,
        'max_steps': 50000,
        'train_tolerance': 1e-3
    })

    trials, best, params = cross_validate(dataset=dataset,
                                          settings=settings,
                                          n_folds=N_FOLDS,
                                          root_folder=CV_ROOT_FOLDER,
                                          n_trials=CV_TRIALS,
                                          search_space=search_space)

    stats = trials.best_trial['results']['averaged']
    return stats, params


if __name__ == '__main__':
    perform_cv(Datasets.TITANIC, TitanicSettings)
