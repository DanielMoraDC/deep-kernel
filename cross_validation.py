import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from model import DeepKernelModel
from protodata.utils import get_data_location, get_logger


logger = get_logger(__name__)


def evaluate(dataset, settings, n_folds, **params):
    """ Returns the average metric over the folds for the
    given execution parameters """
    folds_set = range(n_folds)
    results = []

    for val_fold in folds_set:
        model = DeepKernelModel(verbose=False)
        best_model = model.fit(
            data_settings_fn=settings,
            training_folds=[x for x in folds_set if x != val_fold],
            validation_folds=[val_fold],
            data_location=get_data_location(dataset, folded=True),
            **params
        )

        results.append(best_model)

    avg_results = average_results(results)

    logger.info(
        'Cross validating on: {} \n'.format(params) +
        'Got results: {} \n'.format(avg_results) +
        '----------------------------------------'
    )

    return {
        'loss': -avg_results['val_acc'],
        'averaged': avg_results,
        'parameters': params,
        'all': results,
        'status': STATUS_OK
    }


def cross_validate(dataset,
                   settings,
                   n_folds,
                   n_trials,
                   search_space):

    trials = Trials()
    best = fmin(
        fn=lambda x: evaluate(dataset, settings, n_folds, **x),
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
