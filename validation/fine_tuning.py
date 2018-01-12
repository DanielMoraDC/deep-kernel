import collections
import logging

from training.fit import DeepNetworkTraining

from protodata.utils import get_data_location

logger = logging.getLogger(__name__)


class FineTuningType(object):

    ExtraEpoch = collections.namedtuple(
        'ExtraEpochRefining', ['epochs']
    )

    ExtraLayerwise = collections.namedtuple(
        'ExtraLayerwise', ['epochs_per_layer', 'policy']
    )


def fine_tune_training(dataset,
                       settings_fn,
                       run_folder,
                       fine_tune,
                       num_layers,
                       **params):

    model = DeepNetworkTraining(
        settings_fn=settings_fn,
        data_location=get_data_location(dataset, folded=True),
        folder=run_folder
    )

    last_epoch = params['train_epochs'][-1]

    if isinstance(fine_tune, FineTuningType.ExtraLayerwise):

        logger.info(
            'Layerwise fine-tuning: training %d' % fine_tune.epochs_per_layer
            + ' epochs per layer using %s policy' % fine_tune.policy
        )

        switches = [
            last_epoch + fine_tune.epochs_per_layer * i
            for i in range(1, num_layers)
        ]

        return model.fit(
            num_layers=num_layers,
            max_epochs=last_epoch + fine_tune.epochs_per_layer * num_layers,
            switch_epochs=switches,
            switch_policy=fine_tune.policy,
            restore_folder=run_folder,
            restore_layers=[x for x in range(1, num_layers+1)],
            **params
        )

    elif isinstance(fine_tune, FineTuningType.ExtraEpoch):

        logger.info(
            'Traditional fine-tuning: training %d' % fine_tune.epochs
            + ' extra epochs for the whole network'
        )

        return model.fit(
            num_layers=num_layers,
            max_epochs=last_epoch + fine_tune.epochs,
            restore_folder=run_folder,
            restore_layers=[x for x in range(1, num_layers+1)],
            **params
        )

    else:
        raise ValueError('Unknown refining type {}'.format(fine_tune))
