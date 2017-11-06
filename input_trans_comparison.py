from random import sample
from string import ascii_uppercase
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


n = 64
k = 4

experiment_matrix = {
    LTFArray.transform_random: [2,            50, 100,      200,      400, 600,      1000, 2000],
}

sample_size = 1000
experiments = []
log_name =  ''.join(sample(list(ascii_uppercase), 5))

seed_instance = 0xc0ffee
seed_model = seed_instance + 1

for _ in range(sample_size):
    for transformation in experiment_matrix:
        for N in experiment_matrix[transformation]:
            seed_instance += 2
            seed_model += 2
            experiments.append(
                ExperimentLogisticRegression(
                    log_name=log_name,
                    n=n,
                    k=k,
                    N=N*1000,
                    seed_instance=seed_instance,
                    seed_model=seed_model,
                    transformation=transformation,
                    combiner=LTFArray.combiner_xor,
                    seed_challenge=seed_model,
                    seed_chl_distance=seed_model,
                )
            )

e = Experimenter(log_name, experiments)
e.run()
