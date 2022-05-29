import numpy as np
from numpy import genfromtxt

from dataclasses import dataclass, field
from typing import List, Tuple

from nutrinator.generator import Generator
from nutrinator.generator import GeneratorConfig as GenConf
from nutrinator.nutrionator import Nutrinator

@dataclass
class BeesConfig:
    n: int = 20
    m: int = 5
    e: int = 2
    nsp: int = 3
    nep: int = 5


def stop_criterion(i):
    return i < 10


def join_results(*results: List[tuple]):
    return np.vstack(tuple(A for A, _ in results)), \
           np.vstack(tuple(P for _, P in results))


def generate_new_results(recipes: np.ndarray, portions: np.ndarray,
                         result_evals: List[float],
                         generator: Generator,
                         conf: BeesConfig = BeesConfig()):
    results_indices = np.argsort(result_evals)
    result_recipes, result_portions = recipes[results_indices[::-1]], portions[results_indices[::-1]]

    e_result_recipes, e_result_portions = result_recipes[:conf.e, ...], result_portions[:conf.e, ...]
    m_result_recipes, m_result_portions = result_recipes[conf.e:conf.m, ...], result_portions[conf.e:conf.m, ...]

    # n_nep + n_nsp + n_rand = n
    n_nep, n_nsp, n_rand = conf.nep * conf.e,  \
                           conf.nsp * conf.m - conf.nep * conf.e,  \
                           conf.n - conf.m*conf.nsp

    nep_neighbouring_results = generator.generate_neighbours(e_result_recipes, e_result_portions, n_nep)
    nsp_neighbouring_results = generator.generate_neighbours(m_result_recipes, m_result_portions, n_nsp)
    random_results = generator.generate_days(n_rand)

    return join_results(nsp_neighbouring_results, nep_neighbouring_results, random_results)


# n - number of initial results
# m - number of good bees
# e - number of elite bees
# d - number of days
def bees_algorithm(generator: Generator, fitted_nutrinator: Nutrinator, b_conf: BeesConfig = BeesConfig()):
    recipes, portions = generator.generate_days(b_conf.n)
    result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in zip(recipes, portions)]
    i = 0
    while stop_criterion(i):
        recipes, portions = generate_new_results(recipes, portions, result_evaluations, generator, b_conf)
        result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in zip(recipes, portions)]
        print(len(recipes))
        i += 1


if __name__ == "__main__":
    days = 7
    dishes_per_day = 3

    nutrients_of_recipes = genfromtxt("csv/nutrients.csv", delimiter=",")
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
