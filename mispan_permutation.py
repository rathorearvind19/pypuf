import numpy as np


RESULT_TYPE = np.int8


def swap_every_bit(challenge):
    # Calculate challenges by separately swapping every bit of given challenge
    n = len(challenge)
    c_swapped = np.tile(challenge, (n, 1))
    np.fill_diagonal(c_swapped, c_swapped.diagonal() * -1)
    return c_swapped


def create_swapped_challenges(challenges):
    # Calculate all challenges with hamming distance to given challenges equals 1
    num, n = np.shape(challenges)
    challenges_swapped = np.empty([num, n, n], dtype=RESULT_TYPE)
    for i in range(num):
        challenges_swapped[i] = swap_every_bit(challenges[i])
    return challenges_swapped


def calc_delta(probs):
    # Calculate delta := mean + sd of differences of consecutive output transition probabilities
    n = len(probs)
    diffs = np.empty(n - 1)
    for i in range(n - 1):
        diffs[i] = probs[i+1] - probs[i]
    return np.mean(diffs) + np.std(diffs)


def calc_probs(instance, challenges, permutation):
    """ Calculate the output transition probability (actually frequency) for each index
    over all given challenges on given instance using given permutation """
    probs = np.zeros(instance.n)
    permuted_cs = challenges[:, permutation]
    permuted_cs_swapped = create_swapped_challenges(challenges)[:, :, permutation]
    responses = instance.eval(permuted_cs)
    for i, r in enumerate(responses):
        responses_swapped = instance.eval(permuted_cs_swapped[i])
        for j, r_swapped in enumerate(responses_swapped):
            probs[j] += abs(r - r_swapped) / 2
    return probs / len(challenges)


def permutation_is_good(probs, delta):
    # Check if the permutation is accurate by applying delta as condition
    n = len(probs)
    counter = 0
    for i in range(n - 1):
        if probs[i + 1] - probs[i] <= delta:
            counter += 1
    print(counter)
    if counter > 2:
        return False
    return True


def find_good_permutation(instance, challenges, prng):
    # Determine a good permutation in the sense of Mispan et. al. (target := 0)
    tries = 1000
    for _ in range(tries):
        permutation = prng.permutation(instance.n)
        probs = calc_probs(instance, challenges, permutation)
        delta = calc_delta(probs)
        if permutation_is_good(probs, delta):
            return permutation
    return None
