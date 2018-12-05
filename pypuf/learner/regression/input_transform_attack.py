"""
This module provides an attack on XOR Arbiter PUFs with random but fixed input permutation.
"""
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import ChallengeResponseSet


class InputTransformAttack(Learner):

    def __init__(self, n, k, training_set, validation_set, transform, seed=None,
                 lr_iteration_limit=1000, logger=None, bias=False):
        self.prng = RandomState(seed)
        self.n = n
        self.k = k
        self.transform = transform
        self.training_set = ChallengeResponseSet(
            challenges=self.transform(training_set.challenges, k),
            responses=training_set.responses
        )
        self.validation_set = ChallengeResponseSet(
            challenges=self.transform(validation_set.challenges, k),
            responses=validation_set.responses
        )
        self.logger = logger
        self.bias = bias
        self.lr_learner = LogisticRegression(
            t_set=self.training_set,
            n=n,
            k=k,
            transformation=LTFArray.transform_none,
            combiner=LTFArray.combiner_xor,
            weights_mu=0,
            weights_sigma=1,
            weights_prng=RandomState(self.prng.randint(2**32)),
            logger=logger,
            iteration_limit=lr_iteration_limit,
            bias=False
        )
        assert validation_set.N >= 1000, 'Validation set should contain at least 1000 challenges.'
        self.model = None

    def learn(self):
        # TODO
        self.model = self.lr_learner.learn()
        self.model.transform = self.transform
        return self.model
