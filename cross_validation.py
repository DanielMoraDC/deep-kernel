import numpy as np
import os
import time
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

from model import DeepKernelModel
from protodata.utils import get_data_location, get_logger


logger = get_logger(__name__)


def evaluate(dataset, settings, **params):
    """
    Returns the average metric over the folds for the
    given execution parameters
    """
    n_folds = settings(get_data_location(dataset, folded=True)).get_fold_num()
    folds_set = range(n_folds)
    results = []

    logger.info('Starting evaluation ...')

    for val_fold in folds_set:
        model = DeepKernelModel(verbose=False)
        best_model = model.fit(
            data_settings_fn=settings,
            training_folds=[x for x in folds_set if x != val_fold],
            validation_folds=[val_fold],
            data_location=get_data_location(dataset, folded=True),
            **params
        )

        logger.info(
            'Using validation fold {}: {}'.format(val_fold, best_model)
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
                   n_trials,
                   search_space):
    trials = Trials()
    best = fmin(
        fn=lambda x: evaluate(dataset, settings, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=n_trials,
        trials=trials
    )
    return trials, space_eval(search_space, best)


def average_results(results):
    """ Returns the average of the metrics for all the folds """
    return {
        k: np.mean([x[k] for x in results])
        if k != 'epoch' else np.median([x[k] for x in results])
        for k in results[0].keys()
    }


def evaluate_model(dataset,
                   settings,
                   search_space,
                   output_folder,
                   cv_trials,
                   runs=10,
                   test_batch_size=1):

    trials, params = cross_validate(dataset=dataset,
                                    settings=settings,
                                    n_trials=cv_trials,
                                    search_space=search_space)
    stats = trials.best_trial['result']['averaged']

    # Remove not used parameters
    del params['max_epochs']
    del params['strip_length']

    logger.info('Using model {} for training with results {}'
                .format(params, stats))

    model = DeepKernelModel(verbose=False)

    total_stats = []
    for i in range(runs):

        # Train model for current simulation
        run_folder = os.path.join(output_folder, str(_get_millis_time()))
        logger.info('Running training [{}] in {}'.format(i, run_folder))

        before = time.time()
        model.fit(
            data_settings_fn=settings,
            folder=run_folder,
            max_epochs=stats['epoch'],
            data_location=get_data_location(dataset, folded=True),
            **params
        )
        diff = time.time() - before

        # Evaluate test for current simulation
        test_params = params.copy()
        del test_params['batch_size']
        test_stats = model.predict(
            data_settings_fn=settings,
            folder=run_folder,
            batch_size=test_batch_size,
            data_location=get_data_location(dataset, folded=True),
            **test_params
        )

        test_stats.update({'time(s)': diff})
        logger.info('Training [{}] got results {}'.format(i, test_stats))

        total_stats.append(test_stats)

    return total_stats


def _get_millis_time():
    return int(round(time.time() * 1000))
