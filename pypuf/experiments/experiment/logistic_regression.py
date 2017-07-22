import time
from numpy.random import RandomState
from numpy.linalg import norm
from numpy import array, append
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class ExperimentLogisticRegression(Experiment):
    """
        This Experiment uses the logistic regression learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, n, k, crp_count, seed_instance, seed_model, transformation, combiner,
                 restarts=float('inf'),
                 convergence=1.1):
        self.log_name = log_name
        self.n = n
        self.k = k
        self.N = crp_count
        self.seed_instance = seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.restarts = restarts
        self.convergence = convergence
        self.combiner = combiner
        self.transformation = transformation
        super().__init__(self.log_name, None)
        self.min_dist = 0
        self.test_dist = 0
        self.accuracy = array([])
        self.training_times = array([])
        self.iterations = array([])
        self.dist = 1.0
        self.model = None

    def name(self):
        return 'ExperimentLogisticRegression n={0} k={1} N={2} ' \
               'seed_instance={3}, seed_model={4}'.format(self.n, self.k,
                                                          self.N,
                                                          self.seed_instance,
                                                          self.seed_model)

    def output_string(self):
        return '\n'.join([
            # seed_instance  seed_model i      n      k      N      trans  comb   iter   time   accuracy  model values
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%i\t' '%s\t' '%s\t' '%i\t' '%f\t' '%f\t'    '%s'% (
                self.seed_instance,
                self.seed_model,
                i,
                self.n,
                self.k,
                self.N,
                self.transformation.__name__,
                self.combiner.__name__,
                self.iterations[i],
                self.training_times[i],
                self.accuracy[i],
                ','.join(map(str, self.model.weight_array.flatten() * norm(self.model.weight_array.flatten())))
            )
            for i in range(self.restarts)
        ])

    def learn(self):
        """
            This method learns one instance self.restarts times or a self.convergence threshold is reached.
            The results are saved in:
                self.training_times
                self.dist
                self.accuracy
                self.iterations
        :return:
        """
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.learner = LogisticRegression(
            tools.TrainingSet(instance=self.instance, N=self.N),
            self.n,
            self.k,
            transformation=self.transformation,
            combiner=self.combiner,
            weights_prng=self.model_prng
        )

        i = 0.0
        while i < self.restarts and 1.0 - self.dist < self.convergence:
            start = time.time()
            self.model = self.learner.learn()
            end = time.time()
            self.training_times = append(self.training_times, end - start)
            self.dist = tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.n))
            self.accuracy = append(self.accuracy, 1.0 - self.dist)
            self.iterations = append(self.iterations, self.learner.iteration_count)
            i += 1

    def analyze(self):
        pass
