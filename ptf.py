import sys
from random import sample
from string import ascii_uppercase
from numpy import array
from numpy.random import RandomState, normal, shuffle
from itertools import product
from datetime import datetime, timedelta
from pypuf.experiments.experiment.base import Experiment
from pypuf.experiments.experimenter import Experimenter


class PTFMonomialsExperiment(Experiment):

    @staticmethod
    def monomials_random(n, k, random_instance):
        return random_instance.choice((0, 1), (k, n, n))

    @staticmethod
    def monomials_atf(n, k, random_instance):
        return array([
            [
                [0] * (i - 1) + [1] * (n - i + 1)
                for i in range(1, n+1)
            ]
            for _ in range(k)
        ])

    def __init__(self, log_name, n, k, monomials, seed):
        super().__init__(
            log_name='%s.0x%x_%i_%i_%s' % (
                log_name,
                seed,
                n,
                k,
                monomials.__name__,
            ),
        )
        self.n = n
        self.k = k
        self.seed = seed
        self.random_instance = RandomState(seed)
        self.monomials = monomials
        self.S_coll = monomials(n, k, self.random_instance)

    def run(self):
        S_coll = self.S_coll
        n = self.n
        k = self.k

        # random weights (k*n many)
        w_coll = normal(size=(k,n))

        # compute PTF self.monomials by expanding the product
        # prod_(l=1)^k sum_(i=1)^n w_(l,i) chi_(S_(l,i))
        # where chi_(S_(l,i)) is the product of the input coordiantes with indices that occur in S_(l,i)
        self.ptf_monomials = {}
        c = 0
        start_time = datetime.now()
        for indices in product(range(n), repeat=k):
            #if (c + 1) % 50000 == 0:
            #    progress = c / n**k
            #    elapsed_time = datetime.now() - start_time
            #    remaining_time = timedelta(seconds=(elapsed_time / progress).total_seconds() // 15 * 15) if progress > 0 else '???'
            #    sys.stdout.write("\rprogress: %.4f (%s/%s) remaining time: %s                 " % (c/(n**k), c, n**k, remaining_time))
            c += 1
            #print("Combination %s:" % (indices,))
            running_s = [0] * n  #zeros(n, dtype='int8')
            running_w = 1
            for l in range(k):
                running_s = (running_s + S_coll[l][indices[l]]) % 2
                running_w *= w_coll[l][indices[l]]
                #print("    s: %s, w: %.3f" % (S_coll[l][indices[l]], w_coll[l][indices[l]]))

            #print('total: %s, %f' % (running_s, running_w))
            #print()

            running_s.flags.writeable = False
            monomial_key = running_s.data.tobytes()
            if monomial_key in self.ptf_monomials:
                self.ptf_monomials[monomial_key] += running_w
            else:
                self.ptf_monomials[monomial_key] = running_w

        #for hash in sorted(self.monomials):
        #    print('%s: %.3f' % (hash, self.monomials[hash]))

    def ptf_length(self):
        return len(self.ptf_monomials)

    def ptf_length_relative(self):
        return self.ptf_length() / (2**self.n)

    def analyze(self):
        assert self.ptf_monomials is not None

        self.result_logger.info(
            # seed   n      k      monomials abslen rellen    time
            '0x%x\t' '%i\t' '%i\t' '%s\t'    '%i\t' '%.12f\t' '%.12f\t',
            self.seed,
            self.n,
            self.k,
            self.monomials.__name__,
            self.ptf_length(),
            self.ptf_length_relative(),
            self.measured_time,
        )

def main(args):

    sufficient_k = [0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, \
                    6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, \
                    9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11 ]

    master_seed = 0xbadeaffe
    sample_size = 100
    n_max = 16
    experiments = []
    log_name = ''.join(sample(list(ascii_uppercase), 5))

    seed = master_seed
    for n in range(2, n_max):
        for _ in range(sample_size):
            for monomials in [
                PTFMonomialsExperiment.monomials_atf,
                PTFMonomialsExperiment.monomials_random,
            ]:
                experiments.append(
                    PTFMonomialsExperiment(
                        log_name=log_name,
                        n=n,
                        k=sufficient_k[n],
                        monomials=monomials,
                        seed=seed,
                    )
                )
                seed += 1

    shuffle(experiments)
    Experimenter(log_name, experiments).run()


if __name__ == '__main__':
    main(sys.argv)
