import numpy as np
import os
import time
from hyperopt import fmin, rand, Trials, STATUS_OK, space_eval
import tempfile
import shutil
import logging

from training.fit_validate import DeepNetworkValidation
from training.fit import DeepNetworkTraining

from protodata.utils import get_data_location


logger = logging.getLogger(__name__)


def tune_model(dataset,
               settings_fn,
               search_space,
               n_trials,
               cross_validate,
               folder=None,
               runs=10,
               test_batch_size=1,
               seed=None):
    """
    Tunes a model on training and returns stats of the model on test after
    averaging several runs.
    """
    validate_fn = _cross_validate if cross_validate else _simple_evaluate

    trials = Trials()
    best = fmin(
        fn=lambda x: validate_fn(dataset, settings_fn, **x),
        algo=rand.suggest,  # tpe.suggest for Tree Parzen Window search
        space=search_space,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.RandomState(seed)
    )

    params = space_eval(search_space, best)
    stats = trials.best_trial['result']['averaged']

    # Replace max epochs by once found in the test
    if 'max_epochs' in params:
        del params['max_epochs']
    params['max_epochs'] = stats['epoch']

    logger.info('Using model {} for training with results {}'
                .format(params, stats))

    return _run_setting(dataset=dataset,
                        settings_fn=settings_fn,
                        best_params=params,
                        n_runs=runs,
                        folder=folder,
                        test_batch_size=test_batch_size)


def _run_setting(dataset,
                 settings_fn,
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

    total_stats = []
    for i in range(n_runs):

        # Train model for current simulation
        run_folder = os.path.join(out_folder, str(_get_millis_time()))
        logger.info('Running training [{}] in {}'.format(i, run_folder))

        run_stats = {}

        model = DeepNetworkTraining(
            folder=run_folder,
            settings_fn=settings_fn,
            data_location=get_data_location(dataset, folded=True)
        )

        before = time.time()
        _, fit_loss, fit_error, fit_l2, = model.fit(
            **best_params
        )
        diff = time.time() - before

        run_stats.update({
            'train_loss': fit_loss,
            'train_error': fit_error,
            'train_l2': fit_l2,
            'time(s)': diff
        })

        # Evaluate test for current simulation
        test_params = best_params.copy()
        del test_params['batch_size']
        test_stats = model.predict(
            batch_size=test_batch_size,
            **test_params
        )

        run_stats.update(test_stats)

        logger.info('Training [{}] got results {}'.format(i, run_stats))
        total_stats.append(run_stats)

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

    model = DeepNetworkValidation(
         settings_fn,
         data_location,
         folder=params.get('folder')
    )

    best = model.fit(
        train_folds=[x for x in range(n_folds) if x != validation_fold],
        val_folds=[validation_fold],
        **params
    )

    logger.info('Finished evaluation on {}'.format(params))
    logger.info('Obtained results {}'.format(best))

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
    n_folds = settings_fn(dataset_location).get_fold_num()
    folds_set = range(n_folds)
    results = []

    logger.debug('Starting evaluation on {} ...'.format(params))

    model = DeepNetworkValidation(
         settings_fn,
         dataset_location,
         folder=params.get('folder')
    )

    for val_fold in folds_set:

        best = model.fit(
            train_folds=[x for x in folds_set if x != val_fold],
            val_folds=[val_fold],
            **params
        )

        logger.debug(
            'Using validation fold {}: {}'.format(val_fold, best)
        )

        results.append(best)

    avg_results = _average_results(results)

    logger.info(
        'Finished cross validaton on: {} \n'.format(params)
     )

    logger.info(
        'Results: {} \n'.format(avg_results)
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
    avg = {}
    for k in results[0].keys():
        if k == 'epoch':
            avg[k] = np.median([x[k] for x in results])
        else:
            avg[k] = np.mean([x[k] for x in results])
    return avg


def _get_millis_time():
    return int(round(time.time() * 1000))
