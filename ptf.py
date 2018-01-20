import sys
from numpy import array, ndarray
from numpy.random import RandomState, normal
from itertools import product
from datetime import datetime, timedelta


class PTFMonomials():

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

    def __init__(self, n, k, monomials, random_instance=None):

        self.n = n
        self.k = k
        self.random_instance = random_instance

        # use dead beef by default
        if random_instance is None:
            random_instance = RandomState(0xdeadbeef)

        # random sets S for each LTF input bit (k*n many)
        #S_coll = random_instance.choice((0, 1), (k,n,n))
        S_coll = monomials(n, k, random_instance)

        # random weights (k*n many)
        w_coll = normal(size=(k,n))

        # compute PTF self.monomials by expanding the product
        # prod_(l=1)^k sum_(i=1)^n w_(l,i) chi_(S_(l,i))
        # where chi_(S_(l,i)) is the product of the input coordiantes with indices that occur in S_(l,i)
        self.monomials = {}
        c = 0
        start_time = datetime.now()
        for indices in product(range(n), repeat=k):
            if (c + 1) % 50000 == 0:
                progress = c / n**k
                elapsed_time = datetime.now() - start_time
                remaining_time = timedelta(seconds=(elapsed_time / progress).total_seconds() // 15 * 15) if progress > 0 else '???'
                sys.stdout.write("\rprogress: %.4f (%s/%s) remaining time: %s                 " % (c/(n**k), c, n**k, remaining_time))
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
            hash = running_s.data.tobytes()
            if hash in self.monomials:
                self.monomials[hash] += running_w
            else:
                self.monomials[hash] = running_w

        #for hash in sorted(self.monomials):
        #    print('%s: %.3f' % (hash, self.monomials[hash]))

    def ptf_length(self):
        return len(self.monomials)

    def ptf_length_relative(self):
        return self.ptf_length() / (2**self.n)


def main(args):

    sufficient_k = [0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, \
                    6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, \
                    9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11 ]

    relative_length_rand = []
    absolute_length_rand = []
    relative_length_classic = []
    absolute_length_classic = []

    for n in range(2, 15):
        k = sufficient_k[n]
        rand_puf = PTFMonomials(n, k, monomials=PTFMonomials.monomials_random)
        classic_puf = PTFMonomials(n, k, monomials=PTFMonomials.monomials_atf)
        relative_length_rand += [rand_puf.ptf_length_relative()]
        absolute_length_rand += [rand_puf.ptf_length()]
        relative_length_classic += [classic_puf.ptf_length_relative()]
        absolute_length_classic += [classic_puf.ptf_length()]
        sys.stdout.flush()
        print("\r\rn: %i, k: %i, relative length: %.4f (rand) vs %.4f (classic)" % (n, k, rand_puf.ptf_length_relative(), classic_puf.ptf_length_relative()))

    print(relative_length_rand)
    print(absolute_length_rand)
    print(relative_length_classic)
    print(absolute_length_classic)






if __name__ == '__main__':
    main(sys.argv)
