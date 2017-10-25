import numpy as np
import os
import time
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import tempfile
import shutil

from model import DeepKernelModel
from protodata.utils import get_data_location, get_logger


logger = get_logger(__name__)


def _evaluate_cv(dataset, settings, **params):
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

    avg_results = _average_results(results)

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


def evaluate_model_cv(dataset,
                      settings,
                      search_space,
                      output_folder,
                      cv_trials,
                      runs=10,
                      test_batch_size=1):
    """
    Evaluates a model using cross validation. Suited for small datasets
    """
    trials = Trials()
    best = fmin(
        fn=lambda x: _evaluate_cv(dataset, settings, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=cv_trials,
        trials=trials
    )
    
    params = space_eval(search_space, best)
    stats = trials.best_trial['result']['averaged']

    return _evaluate_setting(dataset=dataset,
                             settings=settings,
                             best_stats=stats,
                             best_params=params,
                             n_runs=runs,
                             output_folder=output_folder,
                             test_batch_size=test_batch_size)


def _evaluate(dataset, settings, max_epochs, **params):
    """
    Returns the metrics after an early stop run
    """
    # Get validation fold randomly and use the rest as training folds
    n_folds = settings(get_data_location(dataset, folded=True)).get_fold_num()
    validation_fold = np.random.randint(n_folds)

    logger.info('Starting evaluation ...')

    model = DeepKernelModel(verbose=False)
    best_model = model.fit(
        data_settings_fn=settings,
        max_epochs=max_epochs,
        training_folds=[x for x in range(n_folds) if x != validation_fold],
        validation_folds=[validation_fold],
        data_location=get_data_location(dataset, folded=True),
        **params
    )

    logger.info(
        'Early stopping finished with best model: {}'.format(best_model)
    )

    return {
        'loss': -best_model['val_acc'],
        'stats': best_model,
        'parameters': params,
        'status': STATUS_OK
    }


def evaluate_model(dataset,
                   settings,
                   search_space,
                   max_epochs,
                   output_folder,
                   cv_trials,
                   runs=10,
                   test_batch_size=1):
    """
    Evaluates a model using a single early stopping run for each
    explored setting. Suitable for large datasets.
    """
    trials = Trials()
    best = fmin(
        fn=lambda x: _evaluate(dataset, settings, max_epochs, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=cv_trials,
        trials=trials
    )

    params = space_eval(search_space, best)
    stats = trials.best_trial['result']['stats']

    return _evaluate_setting(dataset=dataset,
                             settings=settings,
                             best_stats=stats,
                             best_params=params,
                             n_runs=runs,
                             output_folder=output_folder,
                             test_batch_size=test_batch_size)


def _evaluate_setting(dataset,
                      settings,
                      best_stats,
                      best_params,
                      n_runs=10,
                      output_folder=None,
                      test_batch_size=1):
    """
    Fits a model with the training set and evaluates it on the test
    for a given number of times. Then returns the summarized metrics
    on the test set
    """
    # Remove not used parameters
    if 'max_epochs' in best_params:
        del best_params['max_epochs']

    if output_folder is None:
        out_folder = tempfile.mkdtemp()
    else:
        out_folder = output_folder

    logger.info('Using model {} for training with results {}'
                .format(best_params, best_stats))

    model = DeepKernelModel(verbose=False)

    total_stats = []
    for i in range(n_runs):

        # Train model for current simulation
        run_folder = os.path.join(out_folder, str(_get_millis_time()))
        logger.info('Running training [{}] in {}'.format(i, run_folder))

        before = time.time()
        model.fit(
            data_settings_fn=settings,
            folder=run_folder,
            max_epochs=best_stats['epoch'],
            data_location=get_data_location(dataset, folded=True),
            **best_params
        )
        diff = time.time() - before

        # Evaluate test for current simulation
        test_params = best_params.copy()
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

    if output_folder is None:
        shutil.rmtree(out_folder)

    return total_stats


def _average_results(results):
    """ Returns the average of the metrics for all the folds """
    return {
        k: np.mean([x[k] for x in results])
        if k != 'epoch' else np.median([x[k] for x in results])
        for k in results[0].keys()
    }


def _get_millis_time():
    return int(round(time.time() * 1000))
