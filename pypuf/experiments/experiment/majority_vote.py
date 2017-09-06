import numpy as np
from numpy.random import RandomState
from pypuf import tools
from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import SimulationMajorityLTFArray, LTFArray


class ExperimentMajorityVoteFindVotes(Experiment):
    def __init__(self, log_name, n, k, challenge_count, seed_instance, seed_instance_noise, transformation,
                combiner, mu, sigma, sigma_noise, seed_challenges, desired_stability, overall_desired_stability,
                bottom, top, iterations, bias=False):
        super().__init__(
            log_name='%s.0xx%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_instance,
                n,
                k,
                challenge_count,
                transformation.__name__,
                combiner.__name__,
            ),
        )
        self.log_name = log_name
        self.n = n
        self.k = k
        self.N = challenge_count
        self.seed_instance = seed_instance
        self.seed_instance_noise = seed_instance_noise
        self.seed_challenges = seed_challenges
        self.transformation = transformation
        self.combiner = combiner
        self.mu = mu
        self.sigma = sigma
        self.sigma_noise = sigma_noise
        self.bias = bias
        self.desired_stability = desired_stability
        self.overall_desired_stability = overall_desired_stability
        self.bottom = bottom
        self.top = top
        self.vote_count = 0
        self.iterations = iterations

    def run(self):
        # Random number generators
        instance_prng = RandomState(self.seed_instance)
        noise_prng = RandomState(self.seed_instance_noise)
        challenge_prng = RandomState(self.seed_challenges)

        # Weight array for the instance which should be learned
        weight_array = LTFArray.normal_weights(self.n, self.k, self.mu, self.sigma, random_instance=instance_prng)

        while self.bottom < self.top:
            self.vote_count = (self.bottom + self.top) // 2 if ((self.bottom + self.top) // 2) % 2 == 1 else ((
                                                                                                                  self.bottom + self.top) // 2) + 1
            # StablePUF = LTFArray(weights, LTFArray.transform_id, LTFArray.combiner_xor)
            MVPUF = SimulationMajorityLTFArray(weight_array, LTFArray.transform_id,
                                               LTFArray.combiner_xor, self.sigma_noise,
                                               random_instance_noise=noise_prng, vote_count=self.vote_count)

            challenges = np.array(list(tools.random_inputs(self.n, self.N, random_instance=challenge_prng)))
            eval_array = np.zeros(len(challenges))

            for i in range(self.iterations):
                eval_array = eval_array + MVPUF.eval(challenges)

            stab_array = (np.abs(eval_array) + self.iterations) / (2 * self.iterations)
            num_goal_fulfilled = 0
            for i in range(self.N):
                if stab_array[i] >= self.desired_stability:
                    num_goal_fulfilled += 1
            overall_stab = num_goal_fulfilled / self.N

            #     overall_stab      vote_count
            msg = '%f\t'            '%i\t' % (overall_stab, self.vote_count)

            if overall_stab > self.overall_desired_stability:
                self.top = self.vote_count - 1
            else:
                self.bottom = self.vote_count + 1

            self.progress_logger.info(msg)

    def analyze(self):
        #     seed_instance seed_model seed_challenges n      k      N      vote_count  measured_time
        msg = '0x%x\t'      '0x%x\t'   '0x%x\t'        '%i\t' '%i\t' '%i\t' '%i\t'      '%f\t' % (
            self.seed_instance,
            self.seed_model,
            self.seed_challenges,
            self.n,
            self.k,
            self.N,
            self.vote_count,
            self.measured_time
        )

        self.result_logger.info(msg)
