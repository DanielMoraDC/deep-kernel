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

# TODO: reuse code into functions or classes


def _evaluate_layerwise(dataset,
                        settings,
                        folder,
                        max_layers=5,
                        layer_progress_thresh=0.1,
                        **params):
    """
    Incremental layerwise training where we train the latest added layer and
    we keep fixed the previous ones. We stop adding layers using an early
    stop strategy
    """
    n_folds = settings(get_data_location(dataset, folded=True)).get_fold_num()

    all_stats, stop, prev_val_error = [], False, float('inf')
    for i in range(1, max_layers+1):

        logger.info('\n Using %d layers...' % i)

        subfolder = os.path.join(folder, 'layer_%d' % i)
        create_dir(subfolder)

        current_params = params.copy()

        # For layers bigger than one we look for weights in the previous one
        if i > 1:
            current_params['prev_layer_folder'] = os.path.join(
                folder, 'layer_' + str(i-1)
            )

        model = DeepKernelModel()
        stats = model.fit_and_validate(
            layerwise_training=True,
            num_layers=i,
            folder=subfolder,
            **current_params
        )

        logger.info(
            '\nNetwork with {} layers results {} \n'.format(i, stats)
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
            best = {'layer': i-2, 'stats': all_stats[i-2]}
            break
        else:
            prev_val_error = stats['val_error']
            best = {'layer': i, 'stats': stats}

    logger.info('Best configuration found for {}'.format(best))
    return best


def _folded_evaluation(dataset, settings, fit_func, **params):
    """
    Returns the average metric over the folds for the
    given execution parameters
    """
    n_folds = settings(get_data_location(dataset, folded=True)).get_fold_num()
    folds_set = range(n_folds)
    results = []

    logger.info('Starting evaluation on {} ...'.format(params))

    for val_fold in folds_set:
        best_model = fit_func(
            dataset=dataset,
            settings=settings,
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
        'loss': avg_results['val_error'],
        'averaged': avg_results,
        'parameters': params,
        'all': results,
        'status': STATUS_OK
    }


def layerwise_cv(dataset,
                 settings,
                 search_space,
                 output_folder,
                 cv_trials,
                 layer_progress_thresh=0.1,
                 runs=10,
                 test_batch_size=1):
    """
    Evaluates a model using cross validation, where each single execution
    relies on incremental layerwise training
    """

    def ev_func(x):
        return _folded_evaluation(
            dataset,
            settings,
            _evaluate_layerwise,
            folder=output_folder,
            layer_progress_thresh=layer_progress_thresh,
            **x
        )

    trials = Trials()
    best = fmin(
        fn=ev_func,
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


# TODO: rewrite function
def _evaluate_cv(dataset, settings, **params):
    """
    Returns the average metric over the folds for the
    given execution parameters
    """
    n_folds = settings(get_data_location(dataset, folded=True)).get_fold_num()
    folds_set = range(n_folds)
    results = []

    logger.info('Starting evaluation on {} ...'.format(params))

    for val_fold in folds_set:
        model = DeepKernelModel(verbose=False)
        best_model = model.fit_and_validate(
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
        'loss': avg_results['val_error'],
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

    logger.info('Starting evaluation on {} ...'.format(params))

    model = DeepKernelModel(verbose=True)
    best_model = model.fit_and_validate(
        data_settings_fn=settings,
        max_epochs=max_epochs,
        training_folds=[x for x in range(n_folds) if x != validation_fold],
        validation_folds=[validation_fold],
        data_location=get_data_location(dataset, folded=True),
        **params
    )

    logger.info(
        'Early stopping on: {} \n'.format(params) +
        'Got results: {} \n'.format(best_model) +
        '----------------------------------------'
    )

    return {
        'loss': best_model['val_error'],
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


from protodata import datasets
from protodata.utils import get_data_location
import sys
import shutil

if __name__ == '__main__':

    fit = bool(int(sys.argv[1]))

    folder = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/layerwise_model'

    if fit:

        if os.path.isdir(folder):
            shutil.rmtree(folder)

        layer_wise_validate(
            datasets.Datasets.AUS,
            datasets.AusSettings,
            folder=folder,
            l2_ratio=1e-4,
            lr=1e-4,
            lr_decay=0.5,
            lr_decay_epocs=128,
            memory_factor=2,
            hidden_units=128,
            n_threads=4,
            kernel_size=64,
            kernel_mean=0.0,
            kernel_std=0.1,
            strip_length=5,
            batch_size=16,
        )

    else:

        m = DeepKernelModel(verbose=True)

        res = m.predict(
            data_settings_fn=datasets.MagicSettings,
            #folder=folder,
            data_location=get_data_location(datasets.Datasets.MAGIC, folded=True),  # noqa
            memory_factor=2,
            n_threads=4,
            hidden_units=128,
            kernel_size=64,
            kernel_mean=0.0,
            kernel_std=0.1,
            batch_size=16,
        )

        print('Got results {} for prediction'.format(res))
