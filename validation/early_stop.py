import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStop(object):

    def __init__(self, name, progress_thresh, max_succ_errors):
        self._name = name
        self._train_errors = []
        self._prev_val_error = float('inf')
        self._successive_errors = 1
        self._progress_thresh = progress_thresh
        self._max_succ_errors = max_succ_errors

        self._best = {'val_error': float('inf')}
        self._succ_fails = 0

    def epoch_update(self, train_error):
        self._train_errors.append(train_error)

    def strip_update(self, train_run, val_run, epoch):
        best_found, stop = False, False

        # Track best model at validation
        if self._best['val_error'] > val_run.error():
            logger.debug('[%s, %d] New best found' % (self._name, epoch))

            best_found = True
            self._best = {
                'val_loss': val_run.loss(),
                'val_error': val_run.error(),
                'epoch': epoch,
                'train_loss': train_run.loss(),
                'train_error': train_run.error(),
            }

        # Stop using progress criteria
        train_progress = progress(self._train_errors)
        if train_progress < self._progress_thresh:
            logger.debug(
                '[%s, %d] Stuck in training due to ' % (self._name, epoch) +
                'lack of progress (%f < %f). Halting...'
                % (train_progress, self._progress_thresh)
            )
            stop = True

        # Stop using UP criteria
        if self._prev_val_error < val_run.error():
            self._succ_fails += 1
            logger.debug(
                '[%s, %d] Validation error increase. ' % (self._name, epoch) +
                'Successive fails: %d' % self._succ_fails
            )
        else:
            self._succ_fails = 0

        if self._succ_fails == self._max_succ_errors:
            logger.debug(
                '[%s, %d] Validation error increased ' % (self._name, epoch) +
                'for %d successive times. Halting ...' % self._max_succ_errors
            )
            stop = True

        self._prev_val_error = val_run.error()
        train_errors = self._train_errors.copy()
        self._train_errors = []

        return best_found, stop, train_errors

    def set_zero_fails(self):
        self._succ_fails = 0

    def get_best(self):
        return self._best


def progress(strip):
    """
    As detailed in:
        Early stopping - but when?. Lutz Prechelt (1997)

    Progress is the measure of how much training error during a strip
    is larger than the minimum training error during the strip.
    """
    k = len(strip)
    num = np.sum(strip)
    den = np.min(strip)*k
    return 1000 * ((num/den) - 1) if den != 0.0 else 0.0
