from sys import argv, stderr
import numpy as np
import argparse

from pypuf.simulation.arbiter_based.ltfarray import LTFArray, SimulationMajorityLTFArray
from pypuf import tools

def main():

    parser = argparse.ArgumentParser(usage="Experiment to determine the minimum number of votes "
                                           "required to achieve a desired given stability.\n")
    parser.add_argument("stab_c", help="Desired stability of the challenges.", type=float,
                        choices=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
    parser.add_argument("stab_all", help="Overall desired stability.", type=float,
                        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("n", help="Number of bits per Arbiter chain.", type=int,
                        choices=[8, 16, 24, 32, 48, 64, 128])
    parser.add_argument("k_max", help="Maximum number of Arbiter chains.", type=int)
    parser.add_argument("k_range", help="Number of step size between the number of Arbiter chains", type=int,
                        choices=range(1,6))
    parser.add_argument("s_noise", help="Standard deviation of the noise", type=float)
    parser.add_argument("N", help="Number of challenges to evaluate", type=int, choices=range(10, 10001, 10))
    args = parser.parse_args()

    if args.k_max <= 0:
        stderr.write("Negative maximum number of Arbiter chains")
        quit(1)

    iter = 10

    # perform search for minimum number of votes required for each k
    for l in range(args.k_range, args.k_max + 1, args.k_range):
        # perform binary search over vote count to find minimum number of votes
        bottom = 1
        top = 1001
        # create an n-k XOR Arbiter PUF
        weights = LTFArray.normal_weights(args.n, l)

        while bottom < top:
            vote_count = (bottom + top) // 2 if ((bottom + top) // 2) % 2 == 1 else ((bottom + top) // 2) + 1
            # StablePUF = LTFArray(weights, LTFArray.transform_id, LTFArray.combiner_xor)
            MVPUF = SimulationMajorityLTFArray(weights, LTFArray.transform_id,
                                               LTFArray.combiner_xor, args.s_noise, vote_count=vote_count)

            challenges = np.array(list(tools.random_inputs(args.n, args.N)))
            eval_array = np.zeros(len(challenges))

            for i in range(iter):
                eval_array = eval_array + MVPUF.eval(challenges)


            stab_array = (np.abs(eval_array) + iter) / (2 * iter)
            num_goal_fulfilled = 0
            for i in range(args.N):
                if stab_array[i] >= args.stab_c:
                    num_goal_fulfilled += 1
            overall_stab = num_goal_fulfilled / args.N
            if overall_stab > args.stab_all:
                top = vote_count - 1
            else:
                bottom = vote_count + 1
            # print(vote_count, overall_stab)
        print(l, ",", vote_count)

if __name__ == '__main__':
    main()