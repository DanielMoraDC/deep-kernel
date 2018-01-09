import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

import os
import time
import tempfile
import shutil
import logging

from training.fit_validate import DeepNetworkValidation
from training.fit import DeepNetworkTraining
from validation.fine_tuning import fine_tune_training

from protodata.utils import get_data_location


logger = logging.getLogger(__name__)


def tune_model(dataset,
               settings_fn,
               search_space,
               n_trials,
               cross_validate=False,
               fine_tune=None,
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

    # Replace max epochs by once found in the test
    if 'max_epochs' in params:
        del params['max_epochs']

    params['train_epochs'] = stats['train_epochs']
    params['num_layers'] = stats['num_layers']

    logger.info('Using model {} for training with results {}'
                .format(params, stats))

    return _run_setting(dataset=dataset,
                        settings_fn=settings_fn,
                        best_params=params,
                        n_runs=runs,
                        folder=folder,
                        test_batch_size=test_batch_size,
                        fine_tune=fine_tune)


def _run_setting(dataset,
                 settings_fn,
                 best_params,
                 folder=None,
                 n_runs=10,
                 test_batch_size=1,
                 fine_tune=None):
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

        before = time.time()
        _, fit_loss, fit_error, fit_l2 = _incremental_training(
            dataset, settings_fn, run_folder, **best_params
        )

        if fine_tune is not None:
            _, fit_loss, fit_error, fit_l2 = fine_tune_training(
                dataset, settings_fn, run_folder, fine_tune, **best_params
            )

        diff = time.time() - before

        run_stats = {
            'train_loss': fit_loss,
            'train_error': fit_error,
            'train_l2': fit_l2,
            'time(s)': diff
        }

        # Evaluate test for current simulation
        model = DeepNetworkTraining(
            folder=run_folder,
            settings_fn=settings_fn,
            data_location=get_data_location(dataset, folded=True)
        )

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

    best = _incremental_validation(
        dataset, settings_fn, validation_fold, **params
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

    for val_fold in folds_set:

        best = _incremental_validation(
            dataset, settings_fn, val_fold, **params
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


def _incremental_training(dataset,
                          settings_fn,
                          train_folder,
                          num_layers,
                          train_epochs,
                          **params):
    dataset_location = get_data_location(dataset, folded=True)
    prev_folder = None

    for layer in range(1, num_layers+1):

        epochs_layer = train_epochs[layer - 1]

        # Store in subfolders all trainings except from last one
        if layer == num_layers:
            current_folder = train_folder
        else:
            current_folder = os.path.join(train_folder, 'layer_%d' % layer)

        logger.info(
            '[%d] Training %d epochs in folder %s'
            % (layer, epochs_layer, current_folder)
        )

        model = DeepNetworkTraining(
            settings_fn=settings_fn,
            data_location=dataset_location,
            folder=current_folder
        )

        training_stats = model.fit(
            num_layers=layer,
            train_only=layer,
            max_epochs=epochs_layer,
            switch_epochs=None,
            restore_folder=prev_folder,
            restore_layers=[x for x in range(1, layer)],
            **params
        )

        prev_folder = current_folder

    return training_stats


def _incremental_validation(dataset, settings_fn, val_fold, **params):
    """
    Trains a model incrementally so a new layers is appended at the end of the
    network only if it decreases the generalization error and returns the
    stats for the best setting found
    """
    dataset_location = get_data_location(dataset, folded=True)
    n_folds = settings_fn(dataset_location).get_fold_num()
    folds_set = range(n_folds)

    prev_err, prev_folder = float('inf'), None
    epochs, best = [], None

    tmp_folder = tempfile.mkdtemp()

    for layer in range(1, params.get('max_layers')+1):

        logger.debug(
            '[%d] Starting layerwise incremental training' % layer
        )

        current_folder = os.path.join(tmp_folder, 'layer_%d' % layer)

        model = DeepNetworkValidation(
            settings_fn,
            dataset_location,
            folder=current_folder
        )

        fitted = model.fit(
            train_folds=[x for x in folds_set if x != val_fold],
            val_folds=[val_fold],
            num_layers=layer,
            train_only=layer,
            restore_folder=prev_folder,
            restore_layers=[x for x in range(1, layer)],
            layerwise=False,
            **params
        )

        logger.debug(
            '[{}] Training got: {}'.format(layer, fitted)
        )

        if prev_err > fitted['val_error']:
            # Update previous fit
            prev_err = fitted['val_error']
            prev_folder = current_folder

            # Update best model info
            best = fitted
            best.update({'num_layers': layer})
            epochs.append(best['epoch'])

            logger.debug(
                '[%d] Training improved. Going for next layer...' % layer
            )
        else:
            # Layer did not improve, let's keep layer - 1 layers
            logger.debug(
                '[%d] Training stagnated. Stopping...' % layer
            )
            break

    del best['epoch']
    best.update({'train_epochs': epochs})

    shutil.rmtree(tmp_folder)

    return best


def _average_results(results):
    """
    Returns the average of the metrics for all the folds
    """
    avg = {}
    for k in results[0].keys():
        if k == 'epoch' or k == 'num_layers':
            avg[k] = int(np.median([x[k] for x in results]))
        elif k == 'train_epochs':
            avg[k] = _average_layerwise_epochs([x[k] for x in results])
        else:
            avg[k] = np.mean([x[k] for x in results])
    return avg


def _average_layerwise_epochs(epochs):
    longest = np.max([len(x) for x in epochs])
    medians = []
    for current in range(longest):
        valid_values = []
        for trial in epochs:
            if len(trial) > current:
                valid_values.append(trial[current])

        medians.append(int(np.median(valid_values)))
    return medians


def _get_millis_time():
    return int(round(time.time() * 1000))
