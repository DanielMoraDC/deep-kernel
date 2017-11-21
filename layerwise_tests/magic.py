from hyperopt import hp
import numpy as np
import logging

from protodata import datasets
from model_validation import LayerWiseSingleEvaluation

CV_TRIALS = 2
SIM_RUNS = 2
MAX_EPOCHS = 10000

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    search_space = {
        'batch_size': hp.choice('batch_size', [128]),
        # l1 ratio not present in paper
        'l2_ratio': hp.choice('l2_ratio', [0, 1e-1, 1e-2, 1e-3, 1e-4]),
        'lr': hp.choice('lr', [1e-2, 1e-3, 1e-4]),
        'kernel_size': hp.choice('kernel_size', [64, 128, 256, 512]),
        'kernel_std': hp.choice('kernel_std', [1e-2, 0.1, 0.25, 0.5, 1.0]),
        'hidden_units': hp.choice('hidden_units', [512, 1024, 2048])
    }

    # Fixed parameters
    search_space.update({
        'max_layers': 5,
        'layer_progress_thresh': 0.1,
        'lr_decay': 0.5,
        'lr_decay_epocs': 250,
        'n_threads': 4,
        'memory_factor': 2,
        'max_epochs': MAX_EPOCHS,
        'strip_length': 5,
        'progress_thresh': 0.1
    })

    model_ev = LayerWiseSingleEvaluation(
        dataset=datasets.Datasets.MAGIC,
        settings_fn=datasets.MagicSettings,
        folder='/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/magi',
    )

    stats = model_ev.evaluate(
        search_space=search_space,
        cv_trials=CV_TRIALS,
        n_runs=SIM_RUNS,
        test_batch_size=1
    )

    metrics = stats[0].keys()
    for m in metrics:
        values = [x[m] for x in stats]
        print('%s: %f +- %f' % (m, np.mean(values), np.std(values)))
