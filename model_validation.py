import numpy as np
import os
import time
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import tempfile
import shutil
import abc

from model import DeepKernelModel
from training import progress

from protodata.utils import get_data_location, get_logger, create_dir


logger = get_logger(__name__)


class ModelEvaluation(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, dataset, settings_fn, folder=None):
        self._dataset = dataset
        self._settings_fn = settings_fn
        self._folder = folder

    def evaluate(self,
                 search_space,
                 cv_trials,
                 n_runs,
                 test_batch_size):
        """
        Performs a search over the given parameters using a Parzen Window Tree
        and returns the statistics over the given number of runs on the test
        data
        """
        trials = Trials()
        best = fmin(
            fn=lambda x: self._evaluate(**x),
            algo=tpe.suggest,
            space=search_space,
            max_evals=cv_trials,
            trials=trials
        )

        params = space_eval(search_space, best)
        stats = trials.best_trial['result']['averaged']

        return self._evaluate_setting(
            best_stats=stats,
            best_params=params,
            n_runs=n_runs,
            test_batch_size=test_batch_size
        )

    @abc.abstractmethod
    def _evaluate(self, **params):
        """
        This function receives a sample of the parameter space and returns
        the best model found for that configuration given its custom logic.
        """

    def _evaluate_setting(self,
                          best_stats,
                          best_params,
                          n_runs=10,
                          test_batch_size=1):
        """
        Fits a model with the training set and evaluates it on the test
        for a given number of times. Then returns the summarized metrics
        on the test set
        """
        # Remove not used parameters
        if 'max_epochs' in best_params:
            del best_params['max_epochs']

        if self._folder is None:
            out_folder = tempfile.mkdtemp()
        else:
            out_folder = self._folder

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
                data_settings_fn=self._settings_fn,
                folder=run_folder,
                max_epochs=best_stats['epoch'],
                data_location=get_data_location(self._dataset, folded=True),
                **best_params
            )
            diff = time.time() - before

            # Evaluate test for current simulation
            test_params = best_params.copy()
            del test_params['batch_size']
            test_stats = model.predict(
                data_settings_fn=self._settings_fn,
                folder=run_folder,
                batch_size=test_batch_size,
                data_location=get_data_location(self._dataset, folded=True),
                **test_params
            )

            test_stats.update({'time(s)': diff})
            logger.info('Training [{}] got results {}'.format(i, test_stats))

            total_stats.append(test_stats)

        if self._folder is None:
            shutil.rmtree(out_folder)

        return total_stats


class CVEvaluationBase(ModelEvaluation):

    __metaclass__ = abc.ABCMeta

    """
    TODO: add option for custom folder
    """

    def _evaluate(self, **params):
        """
            Returns the average metric over the folds for the
            given execution setting
        """
        dataset_location = get_data_location(self._dataset, folded=True)
        n_folds = self._settings_fn(dataset_location).get_fold_num()
        folds_set = range(n_folds)
        results = []

        logger.info('Starting evaluation on {} ...'.format(params))

        for val_fold in folds_set:

            best_model = self._evaluate_fn(
                training_folds=[x for x in folds_set if x != val_fold],
                validation_folds=[val_fold],
                **params,  # noqa
            )

            logger.info(
                'Using validation fold {}: {}'.format(val_fold, best_model)
            )

            results.append(best_model)

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

    def _evaluate_fn(self,
                     training_folds,
                     validation_folds,
                     **params):
        """
        Evaluates a model given a partition and a search space and returns the
        best setting found
        """


class LayerWiseCVEvaluation(CVEvaluationBase):

    """
    Evaluates a model under a cross-validation process, where each
    run is under an incremental layerwise strategy which uses early stopping.
    """

    def _evaluate_fn(self,
                     training_folds,
                     validation_folds,
                     max_layers=5,
                     layer_progress_thresh=0.1,
                     **params):
        best = layerwise_evaluation(
            dataset=self._dataset,
            settings_fn=self._settings_fn,
            training_folds=training_folds,
            validation_folds=validation_folds,
            max_layers=max_layers,
            layer_progress_thresh=layer_progress_thresh,
            **params
        )

        stats = best['stats']
        stats.update({'layer': best['layer']})
        return stats


class CVEvaluation(CVEvaluationBase):

    """
    Evaluates a model under a cross-validation process, where each
    run is under an early stopping strategy.
    """

    def _evaluate_fn(self,
                     training_folds,
                     validation_folds,
                     **params):
        model = DeepKernelModel(verbose=False)
        best = model.fit_and_validate(
            data_settings_fn=self._settings_fn,
            training_folds=training_folds,
            validation_folds=validation_folds,
            data_location=get_data_location(self._dataset, folded=True),
            **params
        )

        return {
            'loss': best['val_error'],
            'averaged': best,
            'parameters': params,
            'status': STATUS_OK
        }


class SingleEvaluation(ModelEvaluation):

    """
    Evaluates a model using a single early stop run.
    """

    def _evaluate(self, **params):
        # Get validation fold randomly and use the rest as training folds
        data_location = get_data_location(self._dataset, folded=True)
        n_folds = self._settings_fn(data_location).get_fold_num()
        validation_fold = np.random.randint(n_folds)

        model = DeepKernelModel(verbose=False)
        best = model.fit_and_validate(
            training_folds=[x for x in range(n_folds) if x != validation_fold],
            validation_folds=[validation_fold],
            data_settings_fn=self._settings_fn,
            data_location=get_data_location(self._dataset, folded=True),
            **params
        )

        return {
            'loss': best['val_error'],
            'averaged': best,
            'parameters': params,
            'status': STATUS_OK
        }


class SingleLayerWiseEvaluation(ModelEvaluation):

    """
    Evaluates a model using a single early stop run which relies
    on an incremental layerwise strategy which uses early stopping.
    """

    def _evaluate(self, max_layers, layer_progress_thresh, **params):
        data_location = get_data_location(dataset, folded=True)
        n_folds = self._settings_fn(data_location).get_fold_num()
        validation_fold = np.random.randint(n_folds)

        model = DeepKernelModel(verbose=False)
        best = layerwise_evaluation(
            training_folds=[x for x in range(n_folds) if x != validation_fold],
            validation_folds=[validation_fold],
            dataset=self._dataset,
            settings_fn=self._settings_fn,
            folder=self._folder,
            **params
        )

        all_params = params.copy()
        all_params.update({'layer': best['layer']})

        return {
            'loss': best['stats']['val_error'],
            'averaged': best['stats'],
            'parameters': all_params,
            'status': STATUS_OK
        }


def layerwise_evaluation(dataset,
                         settings_fn,
                         folder=None,
                         max_layers=5,
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

        model = DeepKernelModel(verbose=False)
        stats = model.fit_and_validate(
            data_settings_fn=settings_fn,
            data_location=get_data_location(dataset, folded=True),
            layerwise_training=True,
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

    logger.info('Best configuration found for {} \n'.format(best))
    return best


def _average_results(results):
    print(results)
    median_params = ['epoch', 'layer']
    return {
        k: np.mean([x[k] for x in results])
        if k not in median_params else int(np.median([x[k] for x in results]))
        for k in results[0].keys()
    }


def _get_millis_time():
    return int(round(time.time() * 1000))
