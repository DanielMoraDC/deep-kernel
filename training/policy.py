import abc
from numpy.random import RandomState

import logging


logger = logging.getLogger(__name__)


class LayerPolicy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, num_layers):
        self._num_layers = num_layers
        self._layer_id = self.initial_layer_id()

    def layer(self):
        return self._layer_id

    @abc.abstractmethod
    def name(self):
        """
        Returns the policy string identifier
        """

    @abc.abstractmethod
    def initial_layer_id(self):
        """
        Returns the identifier of the first later to train
        """

    @abc.abstractmethod
    def next_layer_id(self):
        """
        Returns the id of next layer to train
        """

    @abc.abstractmethod
    def cycle_ended(self):
        """
        Whether the current layer is the beginning of a new cycle
        """


class CyclicPolicy(LayerPolicy):
    
    def __init__(self, num_layers, **params):
        super(CyclicPolicy, self).__init__(num_layers)

    def initial_layer_id(self):
        return 1

    def next_layer_id(self):
        next_layer = (self._layer_id + 1) % (self._num_layers + 1)
        self._layer_id = max(next_layer, 1)
        return self._layer_id

    def cycle_ended(self):
        return self._layer_id == 1

    def name(self):
        return 'cyclic'


class InverseCyclingPolicy(LayerPolicy):
    
    def __init__(self, num_layers, **params):
        super(InverseCyclingPolicy, self).__init__(num_layers)

    def initial_layer_id(self):
        return self._num_layers

    def next_layer_id(self):
        self._layer_id = self._num_layers \
            if self._layer_id == 1 else self._layer_id - 1
        return self._layer_id

    def cycle_ended(self):
        return self._layer_id == self._num_layers

    def name(self):
        return 'inv_cyclic'


class RandomPolicy(LayerPolicy):

    def __init__(self, num_layers, policy_seed, **params):
        self._count = 0
        self._seed = policy_seed
        self._state = RandomState(policy_seed)
        super(RandomPolicy, self).__init__(num_layers)

    def _random_layer(self):
        return self._state.randint(1, self._num_layers+1)

    def initial_layer_id(self):
        return self._random_layer()

    def next_layer_id(self):
        self._count = (self._count + 1) % self._num_layers
        self._layer_id = self._random_layer()
        return self._layer_id

    def cycle_ended(self):
        return self._count == 0

    def name(self):
        return 'random'
