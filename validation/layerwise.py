import numpy as np
import os
import time
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import tempfile
import shutil

from model import DeepKernelModel
from training import progress

from protodata.utils import get_data_location, get_logger, create_dir


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
    # TODO change
    validate_fn = _cross_validate if cross_validate else _simple_evaluate

    trials = Trials()
    best = fmin(
        fn=lambda x: validate_fn(dataset, settings_fn, folder, **x),
        algo=tpe.suggest,
        space=search_space,
        max_evals=n_trials,
        trials=trials
    )

    params = space_eval(search_space, best)
    best_trial = trials.best_trial['result']
    stats = best_trial['averaged']

    return _run_setting(dataset=dataset,
                        settings_fn=settings_fn,
                        best_stats=stats,
                        best_params=params,
                        n_runs=runs,
                        folder=folder,
                        test_batch_size=test_batch_size)


# TODO: see if we can reuse
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
        _layerwise_fit(
            data_location=get_data_location(dataset, folded=True),
            settings_fn=settings_fn,
            folder=run_folder,
            num_layers=best_stats['layer'],
            layerwise_epochs=best_stats['epochs'],
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


def _simple_evaluate(dataset, settings_fn, folder, **params):
    """
    Returns the metrics for a single early stopping layerwise run
    """
    subfolder = os.path.join(folder, str(_get_millis_time()))

    # Random validation fold for each iteration
    data_location = get_data_location(dataset, folded=True)
    n_folds = settings_fn(data_location).get_fold_num()
    validation_fold = np.random.randint(n_folds)

    # TODO: see if we can return a single value
    best = _layerwise_evaluation(
        data_location=data_location,
        settings_fn=settings_fn,
        training_folds=[x for x in range(n_folds) if x != validation_fold],
        validation_folds=[validation_fold],
        folder=subfolder,
        **params
    )

    return {
        'loss': best['val_error'],
        'averaged': best,
        'parameters': params,
        'status': STATUS_OK
    }


def _cross_validate(dataset, settings_fn,  **params):
    """
    Returns the average metric over the folds for the
    given execution setting
    """
    dataset_location = get_data_location(dataset, folded=True)
    # n_folds = settings_fn(dataset_location).get_fold_num()
    n_folds = 2
    folds_set = range(n_folds)
    results = []

    logger.info('Starting evaluation on {} ...'.format(params))

    for val_fold in folds_set:

        best = _layerwise_evaluation(
            data_location=dataset_location,
            settings_fn=settings_fn,
            training_folds=[x for x in range(n_folds) if x != val_fold],
            validation_folds=[val_fold],
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


def _layerwise_fit(data_location,
                   settings_fn,
                   folder,
                   num_layers,
                   layerwise_epochs,
                   **params):
    """
    Incrementally builds a network given the fitted parameters and stored
    the resulting model into the given folder
    """
    if num_layers != len(layerwise_epochs):
        raise RuntimeError(
            'Number of layers should match the list of epochs'
        )

    if 'max_epochs' in params:
        del params['max_epochs']

    # Keep all except for last layer into temporary folders
    tmp_folder = tempfile.mkdtemp()

    model = DeepKernelModel(verbose=True)
    layer_and_epoch_zip = list(zip(range(1, num_layers+1), layerwise_epochs))
    for layer, max_epochs in layer_and_epoch_zip:

        current_params = params.copy()

        # Last layer goes into destination folder
        if layer == num_layers:
            dst_folder = folder
        else:
            dst_folder = os.path.join(tmp_folder, 'layer_' + str(layer-1))
            create_dir(dst_folder)

        if layer > 1:
            current_params['prev_layer_folder'] = os.path.join(
                tmp_folder, 'layer_' + str(layer-1)
            )

        model.fit(
            data_settings_fn=settings_fn,
            data_location=data_location,
            num_layers=layer,
            max_epochs=max_epochs,
            folder=dst_folder,
            **current_params
        )

    shutil.rmtree(tmp_folder)


def _layerwise_evaluation(data_location,
                          settings_fn,
                          max_layers,
                          folder=None,
                          layer_progress_thresh=0.1,
                          **params):
    """
    Incremental layerwise training where we train the latest added layer
    and we keep fixed the previous ones. We stop adding layers using an
    early stop strategy.
    """

    if folder is None:
        aux_folder = tempfile.mkdtemp()
    else:
        aux_folder = folder

    all_stats, stop, prev_val_error = [], False, float('inf')
    for i in range(1, max_layers+1):

        logger.info('Using %d layers...' % i)

        subfolder = os.path.join(aux_folder, 'layer_%d' % i)
        create_dir(subfolder)

        current_params = params.copy()

        # For #layers > 1 we look for weights in the previous one
        if i > 1:
            current_params['prev_layer_folder'] = os.path.join(
                aux_folder, 'layer_' + str(i-1)
            )

        model = DeepKernelModel(verbose=True)
        stats = model.fit_and_validate(
            data_settings_fn=settings_fn,
            data_location=data_location,
            num_layers=i,
            folder=subfolder,
            **current_params
        )

        logger.info(
            'Network with {} layers results {} \n'.format(i, stats)
        )

        all_stats.append(stats)

        train_progress = progress([x['train_error'] for x in all_stats])
        if len(all_stats) > 1 and train_progress < layer_progress_thresh:
            logger.info(
                '[Layer %d] Progress %f is lower than threshold %f'
                % (i, train_progress, layer_progress_thresh)
            )
            stop = True

        if stats['val_error'] > prev_val_error:
            logger.info(
                '[Layer %d] Error did not decrease. Halting...' % i
            )
            stop = True

        if stop is True:
            best = {'layer': i-1, 'stats': all_stats[i-1]}
            break
        else:
            prev_val_error = stats['val_error']
            best = {'layer': i, 'stats': stats}

    if folder is None:
        shutil.rmtree(aux_folder)

    logger.info('Best configuration found: {} \n'.format(best))

    # Put all together in depth-one dictionary
    best_final = best['stats'].copy()
    best_final.update({'layer': best['layer']})

    # Replace epoch in best by epochs per layer
    epochs_per_layer = [all_stats[i]['epoch'] for i in range(best['layer'])]
    del best_final['epoch']
    best_final['epochs'] = epochs_per_layer

    logger.info('Modified best configuration: {} \n'.format(best_final))

    return best_final


def _average_results(results):
    median_params = ['layer']
    key_subset = [k for k in results[0].keys() if k != 'epochs']

    averages = {
        k: np.mean([x[k] for x in results])
        if k not in median_params else int(np.median([x[k] for x in results]))
        for k in key_subset
    }

    averages['epochs'] = _average_layerwise_epochs(
        [x['epochs'] for x in results], num_layers=averages['layer']
    )

    return averages


def _average_layerwise_epochs(epochs, num_layers):
    median_epochs = []
    for layer in range(1, num_layers + 1):
        valid_values = []
        for trial in epochs:
            print(trial)
            print(layer)
            if layer <= len(trial):
                valid_values.append(trial[layer-1])
        median_epochs.append(np.median(valid_values))

    return median_epochs


def _get_millis_time():
    return int(round(time.time() * 1000))
