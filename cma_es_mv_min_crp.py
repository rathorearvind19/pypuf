from sys import argv, stdout, stderr
from random import sample
from string import ascii_uppercase
from numpy import median
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.reliability_based_cma_es import ExperimentReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


def determine_minimum_example_size_for_cma_es_success(
        n=64,
        k=8,
        vote_count=1,
        trials=1,
        estimation_accuracy=100,
        accuracy_threshold=.75,
        N_max=2**16,
):
    # initialize variables
    accuracy = .5
    N = 1
    log_file_prefix = ''.join(sample(list(ascii_uppercase), 5))

    def approximate_accuracy(
            N,
            seed_instance = 0x5eed,
            seed_instance_noise = 0xbad,
            seed_model = 0xc0ffee,
            seed_challenges = 0xf000,
    ):
        """Runs one set of experiments and averages the result"""

        # setup stuff
        e = Experimenter(log_file_prefix, [])
        for i in range(trials):
            e.experiments.append(
                ExperimentReliabilityBasedCMAES(
                    log_name=log_file_prefix,
                    n=n,
                    k=k,
                    challenge_count=N,
                    seed_instance=seed_instance,
                    seed_instance_noise=seed_instance_noise,
                    seed_model=seed_model,
                    transformation=LTFArray.transform_id,
                    combiner=LTFArray.combiner_xor,
                    mu=0,
                    sigma=1,
                    sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(n, 1, .5),
                    vote_count=vote_count,
                    repetitions=21,
                    limit_step_size=1 / 2 ** 8,
                    limit_iteration=500,
                    seed_challenges=seed_challenges,
                    bias=False,
                )
            )
            seed_instance += 1
            seed_instance_noise += 1
            seed_model += 1
            seed_challenges += 1

        # run stuff
        result_queue = e.run()
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # collect results
        accuracy = median(results)
        stdout.write('obtained accuracy %1.5f with N=%i\n' % (accuracy, N))
        return accuracy

    # exponentially increase N until we see first success
    stdout.write('starting exponential increase\n')
    while accuracy < accuracy_threshold:
        N *= 2
        accuracy = approximate_accuracy(N)
        if N >= N_max:
            raise Exception('no success even with the maximum allowed number of examples, N_max=%i' % N_max)

    # binary search a value in between N and N/2
    top = N
    bottom = N/2

    stdout.write('\nstarting binary search in between %i and %i\n' % (bottom, top))
    while (top - bottom > estimation_accuracy):

        if accuracy > accuracy_threshold:
            # mark N as new top (that is, this N is sufficient to obtain accuracy_threshold)
            top = N
            stdout.write('decreasing top, ')
        else:
            # mark N as new bottom (that is, this N is not sufficient to obtain accuracy_threshold)
            bottom = N
            stdout.write('increasing bottom, ')

        N = round((top + bottom) / 2)
        stdout.write('N=%i\n' % N)
        accuracy = approximate_accuracy(N)




def main(args):

    if len(args) != 5:
        stderr.write('Experiment to determine the minimum CRPs needed to learn a Majority Vote XOR\n')
        stderr.write('Arbiter PUF with CMA-ES.\n')
        stderr.write('Usage:\n')
        stderr.write('cma_es_mv_min_crps.py n k trials\n')
        stderr.write('               n: number of bits per Arbiter chain\n')
        stderr.write('               k: number of Arbiter chains\n')
        stderr.write('      vote_count: number of votes in the MV XOR Arbiter PUF\n')
        stderr.write('          trials: number of repeated learning attempts to average results\n')
        stderr.write('                  for given parameters\n')
        quit(1)

    n = int(args[1])
    k = int(args[2])
    vote_count = int(args[3])
    trials = int(args[4])

    determine_minimum_example_size_for_cma_es_success(n, k, vote_count, trials)

if __name__ == '__main__':
    main(argv)
