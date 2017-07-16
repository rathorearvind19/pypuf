from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from sys import argv, stdout, stderr
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter
from random import sample
from string import ascii_uppercase


def main(args):

    log_file_prefix = ''.join(sample(list(ascii_uppercase), 5))
    stderr.write('log file prefix: %s\n' % log_file_prefix)

    n = 64
    k = 4
    combiner = LTFArray.combiner_xor

    instance_samples = 10
    experiments = []

    seed_instance = 0x15eed
    seed_model = 0x5eed

    for transformation in [
        LTFArray.transform_1_1_bent,
        LTFArray.transform_1_n_bent,
        LTFArray.transform_atf,
        LTFArray.transform_id,
        LTFArray.transform_lightweight_secure,
        #LTFArray.transform_mm,
        LTFArray.transform_polynomial,
        LTFArray.transform_shift,
        LTFArray.transform_shift_lightweight_secure,
        LTFArray.transform_soelter_lightweight_secure,
    ]:
        for N in [
            12000,
            30016,
            100000,
            #250016,
            #1000000,
        ]:
            for j in range(instance_samples):

                experiments.append(
                    ExperimentLogisticRegression(
                        log_name='%s_instance0x%x_model0x%x.log' % (log_file_prefix, seed_instance, seed_model),
                        n=n,
                        k=k,
                        crp_count=N,
                        seed_model=seed_model,
                        seed_instance=seed_instance,
                        transformation=transformation,
                        combiner=combiner,
                        restarts=1,
                    )
                )

                seed_instance += 1
                seed_model += 1

    Experimenter(log_file_prefix + '.log', experiments).run()

if __name__ == '__main__':
    main(argv)
