import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from nutrinator.generator import Generator
from nutrinator.generator import GeneratorConfig as GenConf
from nutrinator.nutrionator import Nutrinator

@dataclass
class BeesConfig:
    n: int = 10
    m: int = 5
    e: int = 2
    nsp: int = 15
    nep: int = 6


def stop_criterion(i):
    return i < 10


def generate_new_results(results: List[Tuple], result_evals: List[float],
                         generator: Generator,
                         conf: BeesConfig = BeesConfig()):
    best_results_ind = np.argsort(result_evals)

    m_best_results = [results[i] for i in best_results_ind[:conf.m]]
    e_best_results = m_best_results[:conf.e]
    nsp_neighbouring_results = generator.generate_neighbours(m_best_results, conf.nsp)
    nep_neighbouring_results = generator.generate_neighbours(e_best_results, conf.nep)
    random_results = [generator.generate_days() for _ in range(conf.n - conf.m)]

    return m_best_results + \
           nsp_neighbouring_results + nep_neighbouring_results + \
           random_results


# n - number of initial results
# m - number of good bees
# e - number of elite bees
# d - number of days
def bees_algorithm(generator: Generator, fitted_nutrinator: Nutrinator, b_conf: BeesConfig = BeesConfig()):
    results = [generator.generate_days() for _ in range(b_conf.n)]
    result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in results]
    i = 0
    while stop_criterion(i):
        results = generate_new_results(results, result_evaluations, generator, b_conf)
        result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in results]
        print(len(results))
        i += 1


if __name__ == "__main__":
    days = 7
    dishes_per_day = 3

    # TODO load nutrients from a file
    nutrients_of_recipes = np.array([[20, 30, 20, 10],
                                      [40, 50, 20, 20],
                                      [50, 10, 10, 30],
                                      [10, 40, 30, 10],
                                      [40, 50, 20, 20],
                                      [50, 10, 10, 30],
                                      [10, 40, 30, 10]])
    nutrient_demand = np.array([[100, 100, 100, 100],
                                [100, 100, 100, 100],
                                [100, 100, 100, 100],
                                [100, 100, 100, 100],
                                [100, 100, 100, 100],
                                [100, 100, 100, 100],
                                [100, 100, 100, 100]])
    nutrient_special_demand = np.zeros((days, dishes_per_day, 4))

    generator = Generator(GenConf(days=days,
                                  dishes=dishes_per_day,
                                  recipes_size=nutrients_of_recipes.shape[0]))
    nutrinator = Nutrinator(nutrients_of_recipes)
    nutrinator.fit(nutrient_demand, nutrient_special_demand)

    bees_algorithm(generator, nutrinator)