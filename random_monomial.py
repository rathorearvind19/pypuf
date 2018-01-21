import sys
from random import sample
from string import ascii_uppercase
from numpy import array
from numpy.random import RandomState, normal, shuffle
from itertools import product
from datetime import datetime, timedelta
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest
from pypuf.experiments.experimenter import Experimenter
from pypuf.property_test.base import PropertyTest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


sufficient_k = [0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, \
                6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, \
                9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11]


def generate_instances(n, seed):
    k = sufficient_k[n]
    return [
        LTFArray(
            weight_array=LTFArray.normal_weights(n, k, random_instance=RandomState(seed)),
            transform=LTFArray.generate_random_monomial_transform(n, k, seed + i, noise=0),
            combiner=LTFArray.combiner_xor,
        )
        for i in range(50)
    ]


def main(args):

    master_seed = 0xbadeaffe
    sample_size = 200
    n = 32
    experiments = []
    log_name = ''.join(sample(list(ascii_uppercase), 5))

    seed = master_seed
    for i in range(sample_size):
        experiments.append(
            ExperimentPropertyTest(
                log_name=log_name + str(i),
                test_function=PropertyTest.uniqueness_statistic,
                challenge_count=1000,
                measurements=50,
                challenge_seed=seed,
                ins_gen_function=generate_instances,
                param_ins_gen={"n": n, "seed": seed},
            )
        )
        seed += 1000 * sample_size

    shuffle(experiments)
    Experimenter(log_name, experiments).run()


if __name__ == '__main__':
    main(sys.argv)
