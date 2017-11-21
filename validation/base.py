import numpy as np
import os
import time
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import tempfile
import shutil

from model import DeepKernelModel

from protodata.utils import get_data_location, get_logger


logger = get_logger(__name__)


def tune_model(dataset,
               settings_fn,
               search_space,
               n_trials,
               cross_validate,
               folder=None,
               runs=10,
               test_batch_size=1):
    """
    Tunes a model on training and returns stats of the model on test after
    averaging several runs.
    """
    validate_fn = _cross_validate if cross_validate else _simple_evaluate

    trials = Trials()
    best = fmin(
        fn=lambda x: validate_fn(dataset, settings_fn, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=n_trials,
        trials=trials
    )

    params = space_eval(search_space, best)
    stats = trials.best_trial['result']['averaged']

    return _run_setting(dataset=dataset,
                        settings_fn=settings_fn,
                        best_stats=stats,
                        best_params=params,
                        n_runs=runs,
                        output_folder=folder,
                        test_batch_size=test_batch_size)


def _run_setting(dataset,
                 settings_fn,
                 best_stats,
                 best_params,
                 folder=None,
                 n_runs=10,
                 test_batch_size=1):
    """
    Fits a model with the training set and evaluates it on the test
    for a given number of times. Then returns the summarized metrics
    on the test set.
    """
    if folder is None:
        out_folder = tempfile.mkdtemp()
    else:
        out_folder = folder

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
            data_settings_fn=settings_fn,
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
            data_settings_fn=settings_fn,
            folder=run_folder,
            batch_size=test_batch_size,
            data_location=get_data_location(dataset, folded=True),
            **test_params
        )

        test_stats.update({'time(s)': diff})
        logger.info('Training [{}] got results {}'.format(i, test_stats))

        total_stats.append(test_stats)

    if folder is None:
        shutil.rmtree(out_folder)

    return total_stats


def _simple_evaluate(dataset, settings_fn, **params):
    """
    Returns the metrics for a single early stopping run
    """
    data_location = get_data_location(dataset, folded=True)
    n_folds = settings_fn(data_location).get_fold_num()
    validation_fold = np.random.randint(n_folds)

    model = DeepKernelModel(verbose=False)
    best = model.fit_and_validate(
        training_folds=[x for x in range(n_folds) if x != validation_fold],
        validation_folds=[validation_fold],
        data_settings_fn=settings_fn,
        data_location=data_location,
        **params
    )

    return {
        'loss': best['val_error'],
        'averaged': best,
        'parameters': params,
        'status': STATUS_OK
    }


def _cross_validate(dataset, settings_fn, **params):
    """
    Returns the average metric over the folds for the
    given execution setting
    """
    dataset_location = get_data_location(dataset, folded=True)
    # n_folds = self._settings_fn(dataset_location).get_fold_num()
    n_folds = 2
    folds_set = range(n_folds)
    results = []

    logger.info('Starting evaluation on {} ...'.format(params))

    for val_fold in folds_set:

        model = DeepKernelModel(verbose=False)
        best = model.fit_and_validate(
            data_settings_fn=settings_fn,
            training_folds=[x for x in folds_set if x != val_fold],
            validation_folds=[val_fold],
            data_location=get_data_location(dataset, folded=True),
            **params
        )

        logger.info(
            'Using validation fold {}: {}'.format(val_fold, best)
        )

        results.append(best)

    avg_results = _average_results(results)

    logger.info(
        'Cross validating on: {} \n'.format(params) +
        'Got results: {} \n'.format(avg_results) +
        '---------------------------------------- \n'
    )

    return {
        'loss': avg_results['val_error'],
        'averaged': avg_results,
        'parameters': params,
        'all': results,
        'status': STATUS_OK
    }


def _average_results(results):
    """
    Returns the average of the metrics for all the folds
    """
    return {
        k: np.mean([x[k] for x in results])
        if k != 'epoch' else np.median([x[k] for x in results])
        for k in results[0].keys()
    }


def _get_millis_time():
    return int(round(time.time() * 1000))