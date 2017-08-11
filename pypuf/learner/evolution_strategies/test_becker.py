import numpy as np
import itertools as iter
from pypuf import tools
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES as Becker
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray


transform = LTFArray.transform_atf
combiner = LTFArray.combiner_xor

# test for is_different_LTF checked!
new_LTF = np.array([0.2, -1.1, -0.7, -0.9, -0.49, 1.621, 1.6, 0.77])
different_LTFs = np.array([[-1.2, -0.42, -0.81, 1.62, 0.47, 0.23, -0.37, 0.68],
                           [0.61, -2.5, 0.29, 1.16, -1.83, 0.72, -0.82, 0.775]])
num_of_LTFs = 2
n = 8
num = 17
prng = np.random.RandomState(0x1111)
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
challenge_num = num
is_diff = Becker.is_different_LTF(new_LTF, different_LTFs, num_of_LTFs, challenges, transform, combiner)
print('is_diff\n', is_diff)

# test for get_fitness_function checked!
n = 8
num = 16
prng = prng
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
challenge_num = challenge_num
measured_rels = np.array([1.5, 2.5, 2.5, 2.5, 0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5])
fitness_function = Becker.get_fitness_function(challenges, challenge_num, transform, combiner)
print('fitness_function\n', fitness_function)

# test for build_LTFArrays checked!
weight_arrays = np.array([[1,1], [-2,-2]])
built_LTFArrays = Becker.build_LTFArrays(weight_arrays, transform, combiner)
print('built_LTFArrays\n', built_LTFArrays)

# test for get_delay_differences checked!
built_LTFArrays = built_LTFArrays
pop_size = 2
n = 2
num = 4
challenges = np.array(list(tools.sample_inputs(n, num, prng)))
delay_diffs = Becker.get_delay_differences(built_LTFArrays, pop_size, challenges)
print('delay_diffs\n', delay_diffs)

# test for get_reliabilities checked!
delay_diffs = np.array([[0.03, 1.77, -1.48, -2.4],
                        [3.62, 2, 6.2, -1.8],
                        [0.1, -3.2, 2.4, 2.1],
                        [-2.4, 0.6, 4.6, -2.77]])
epsilons = np.array([0.5, -2.1, 4.2, 1.5])
reliabilities = Becker.get_reliabilities(delay_diffs, epsilons)
print('reliabilities\n', reliabilities)

# test for get_correlations checked!
reliabilities = np.array([[1, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1],
                          [0, 1, 0, 0, 0, 1],
                          [1, 0, 1, 0, 1, 1]])
measured_rels = np.array([2.5, 1.5, 2.5, 0.5, 1.5, 2.5])
correlations = Becker.get_correlations(reliabilities, measured_rels)
print('correlations\n', correlations)

# test for set_pole_of_LTFs checked!
different_LTFs = np.array([[-0.1, 0.15, 0.2], [0.1, 0.15, -0.2]])
challenges = np.array([[1,1,1], [-1,1,-1], [-1,-1,1], [1,-1,-1]])
xor_LTFArray = LTFArray(different_LTFs, transform, combiner)
responses_model = xor_LTFArray.eval(challenges)
common_responses = np.array([1, 1, -1, 1])
print('different_LTFs\n', different_LTFs)
polarized_LTFs = Becker.set_pole_of_LTFs(different_LTFs, challenges, common_responses, transform, combiner)
print('polarized_LTFs\n', polarized_LTFs)

# test for is_correlated checked!
responses_new_LTF = np.array([1, 1, 1, 1, 1, -1])
responses_diff_LTFs = np.array([[1, 1, 1, 1, 1, 1],
                                [-1, -1, -1, 1, 1, 1]])
is_same_LTF = Becker.is_correlated(responses_new_LTF, responses_diff_LTFs)
print('is_same_LTF\n', is_same_LTF)

# test for get_common_responses checked!
responses = np.array([[-1, -1, 1, -1, 1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1],
                      [1, -1, 1, -1, 1, -1, 1],
                      [-1, -1, 1, -1, 1, 1, 1],
                      [-1, -1, -1, -1, 1, 1, 1]])
common_responses = Becker.get_common_responses(responses)
print('common_responses\n', common_responses)

# test for get_measured_rels checked!
responses = responses
measured_rels = Becker.get_measured_rels(responses)
print('measured_rels\n', measured_rels)

# test for learn
weight_array = np.array([[0.311, 2.75, -1.82, -0.558, 1.22, 0.5, -0.05, -0.811, -1.2, -0.422, 0.72, 1.681]])
                         #[-1.42, 0.39, -1.491, 0.8, 2.08, -0.333, 0.539, -1.36, 2.51, -1.77, -0.193, 0.81],
                         #[0.73, 1.38, 1.91, -0.7, -3.107, 1.84, 1.014, -0.7203, -0.25, -0.267, 1.17, 0.69],
                         #[-2.3, 0.218, -1.25, -0.412, 0.294, -1.16, -0.18, 0.4, 2.3, -1.37, -0.53, 0.0874]])
n = 12
sigma_weight = 1
noisiness = 0.2
sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, sigma_weight, noisiness)
instance = NoisyLTFArray(weight_array, transform, combiner, sigma_noise)
k = 1
num = 2**11
transform = LTFArray.transform_id
prng = np.random.RandomState(0x1111)
challenges = tools.sample_inputs(n, num, prng)
repeat = 5
responses = np.zeros((repeat, num))
for i in range(repeat):
    challenges, cs = iter.tee(challenges)
    responses[i, :] = instance.eval(np.array(list(cs)))
precision = 1 / 2**10
pop_size = 30
parent_size = 10
priorities = np.array([.20, .15, .12, .10, .08, .07, .07, .07, .07, .07])
prng = np.random.RandomState(0x1234)
challenges = np.array(list(challenges))
becker = Becker(k, n+1, transform, combiner, challenges, responses, repeat,
                precision, pop_size, parent_size, priorities, prng)

xor_LTFArray = becker.learn()
print(vars(xor_LTFArray))