import numpy as np
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import sample_inputs, substitute_aes


# Randomly choose seed
seed = np.random.randint(2**32)
print('seed = ', seed)

# Create PUF instance
n = 32
k = 2
prng_i = np.random.RandomState(seed)
transformation = LTFArray.transform_aes_sbox
combiner = LTFArray.combiner_xor

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n, k, random_instance=prng_i),
    transform=transformation,
    combiner=combiner
)


# Sample challenges
num = 32000
prng_c = np.random.RandomState(seed+1)
challenges = sample_inputs(n, num, prng_c)

responses = instance.eval(challenges)

print(responses)
