from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import sample_inputs
from mispan_permutation import *
import datetime


# Randomly choose seed
seed = np.random.randint(2**32)
print('seed = ', seed)

# Create PUF instance
n = 32
k = 1
prng_i = np.random.RandomState(seed)
transformation = LTFArray.transform_atf
combiner = LTFArray.combiner_xor

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n, k, random_instance=prng_i),
    transform=transformation,
    combiner=combiner
)

time_start = datetime.datetime.now()


# Sample challenges
num = 32000
prng_c = np.random.RandomState(seed+1)
challenges = sample_inputs(n, num, prng_c)
time_sampling = datetime.datetime.now()

"""
# Create swapped challenges
cs_swapped = create_swapped_challenges(challenges)
time_swapping = datetime.datetime.now()


# Permute challenges
prng_p = np.random.RandomState(seed+2)
permutation = prng_p.permutation(n)
permuted_cs = challenges[:, permutation]
permuted_cs_swapped = cs_swapped[:, :, permutation]
time_permuting = datetime.datetime.now()


# Calculate output transition probabilities
responses = instance.eval(permuted_cs)
probs = np.zeros(instance.n)
for i, r in enumerate(responses):
    responses_swapped = instance.eval(permuted_cs_swapped[i])
    for j, r_swapped in enumerate(responses_swapped):
        probs[j] += abs(r - r_swapped) / 2
time_evaluating = datetime.datetime.now()


# Calculate condition delta
delta = calc_delta(probs)
time_delta = datetime.datetime.now()


# Check if permutation is good
check = permutation_is_good(probs, delta)
time_checking = datetime.datetime.now()


times = list()
times.append(time_sampling - time_start)
times.append(time_swapping - time_sampling)
times.append(time_permuting - time_swapping)
times.append(time_evaluating - time_permuting)
times.append(time_delta - time_evaluating)
times.append(time_checking - time_delta)
times.append(time_checking - time_start)

print('\nsampling\t',   times[0], '\t', 100*round(times[0]/times[6], 3), '%',
      '\nswapping\t',   times[1], '\t', 100*round(times[1]/times[6], 3), '%',
      '\npermuting\t',  times[2], '\t', 100*round(times[2]/times[6], 3), '%',
      '\nevaluating\t', times[3], '\t', 100*round(times[3]/times[6], 3), '%',
      '\ndelta\t\t',    times[4], '\t', 100*round(times[4]/times[6], 3), '%',
      '\nchecking\t',   times[5], '\t', 100*round(times[5]/times[6], 3), '%',
      '\n\ntotal\t\t',  times[6], '\t', 100*round(times[6]/times[6], 3), '%')
"""

prng_p = np.random.RandomState(seed+2)
permutation = find_good_permutation(instance, challenges, prng_p)

print(permutation)
