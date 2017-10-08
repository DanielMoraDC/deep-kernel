import time
import os
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from model import DeepKernelModel
from protodata.utils import get_data_location


FIXED_PARAMETERS = {
    'n_threads': 2,
    'memory_factor': 2,
    'summary_steps': 100,
    'validation_interval': 500,
    'max_steps': 50000,
    'train_tolerance': 1e-3
}


def evaluate(dataset, settings, n_folds, root_folder, **params):
    """ Returns the average metric over the folds for the
    given execution parameters """
    execution_folder = os.path.join(root_folder, str(_get_millis_time()))

    print('Logging parameters {} in {}'.format(params, execution_folder))

    folds_set = range(n_folds)

    results = []
    for val_fold in folds_set:

        fold_folder = os.path.join(execution_folder, 'val_fold_%d' % val_fold)

        model = DeepKernelModel()
        best_model = model.fit(
            data_settings_fn=settings,
            folder=fold_folder,
            training_folds=[x for x in folds_set if x != val_fold],
            validation_folds=[val_fold],
            data_location=get_data_location(dataset, folded=True),
            **params
        )

        results.append(best_model)

    avg_results = average_results(results)

    return {
        'loss': -avg_results['val_acc'],
        'averaged': avg_results,
        'all': results,
        'status': STATUS_OK
    }


def cross_validate(dataset,
                   settings,
                   n_folds,
                   root_folder,
                   n_trials,
                   search_space):

    trials = Trials()
    best = fmin(
        fn=lambda x: evaluate(dataset, settings, n_folds, root_folder, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=n_trials,
        trials=trials
    )

    return trials, best, space_eval(search_space, best)


def average_results(results):
    """ Returns the average of the metrics for all the folds """
    return {
        k: np.mean([x[k] for x in results])
        for k in results[0].keys()
    }


def _get_millis_time():
    return int(round(time.time() * 1000))
