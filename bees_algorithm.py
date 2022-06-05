import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import genfromtxt

from dataclasses import dataclass, field
from typing import List, Tuple

from nutrinator.generator import Generator
from nutrinator.generator import GeneratorConfig as GenConf
from nutrinator.nutrionator import Nutrinator


@dataclass
class BeesConfig:
    n: int = 100
    m: int = 40
    e: int = 10
    nsp: int = 3
    nep: int = 5


def stop_criterion(i):
    return i < 10000


def generate_new_results(recipes: np.ndarray, portions: np.ndarray,
                         result_evals: List[float],
                         generator: Generator,
                         conf: BeesConfig = BeesConfig()):

    ## Selecting best and elite sites for neighbourhood search
    results_indices = np.argsort(result_evals)
    result_recipes, result_portions = recipes[results_indices[::-1]], portions[results_indices[::-1]]

    e_result_recipes, e_result_portions = result_recipes[:conf.e, ...], result_portions[:conf.e, ...]
    m_result_recipes, m_result_portions = result_recipes[conf.e:conf.m, ...], result_portions[conf.e:conf.m, ...]

    ## Neighbourhood search
    shape = (generator.config.days, generator.config.dishes)
    nep_neighbouring_recipes = np.empty((conf.e, *shape), dtype=int)
    nep_neighbouring_portions = np.empty((conf.e, *shape), dtype=float)
    nsp_neighbouring_recipes = np.empty((conf.m - conf.e, *shape), dtype=int)
    nsp_neighbouring_portions = np.empty((conf.m - conf.e, *shape), dtype=float)

    # Neighbourhood search for elite sites
    for idx, (recipe, portion) in enumerate(zip(e_result_recipes, e_result_portions)):
        nep_neighbouring_recipes[idx, ...] = recipe
        nep_neighbouring_portions[idx, ...] = portion
        best_fit = nutrinator.compute(recipe, portion)
        for i in range(conf.nep):
            # ma zwracac tupla
            recruited_bee = generator.generate_neighbour(recipe, portion)
            fit = nutrinator.compute(recruited_bee[0], recruited_bee[1])
            if fit < best_fit:
                nep_neighbouring_recipes[idx, ...] = recruited_bee[0]
                nep_neighbouring_portions[idx, ...] = recruited_bee[1]
                best_fit = fit

    # Neighbourhood search for best sites
    for idx, (recipe, portion) in enumerate(zip(m_result_recipes, m_result_portions)):
        nsp_neighbouring_recipes[idx, ...] = recipe
        nsp_neighbouring_portions[idx, ...] = portion
        best_fit = nutrinator.compute(recipe, portion)
        for i in range(conf.nsp):
            recruited_bee = generator.generate_neighbour(recipe, portion)
            fit = nutrinator.compute(recruited_bee[0], recruited_bee[1])
            if fit < best_fit:
                nsp_neighbouring_recipes[idx, ...] = recruited_bee[0]
                nsp_neighbouring_portions[idx, ...] = recruited_bee[1]
                best_fit = fit

    ## Random global search
    random_results = generator.generate_days(conf.n-conf.m)

    best_recipes = np.concatenate((nep_neighbouring_recipes, nsp_neighbouring_recipes, random_results[0]), axis=0)
    best_portions = np.concatenate((nsp_neighbouring_recipes, nsp_neighbouring_portions, random_results[1]), axis=0)

    return best_recipes, best_portions


# n - number of initial results
# m - number of good bees
# e - number of elite bees
# d - number of days
def bees_algorithm(generator: Generator, fitted_nutrinator: Nutrinator, b_conf: BeesConfig = BeesConfig()):
    recipes, portions = generator.generate_days(b_conf.n)
    result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in zip(recipes, portions)]
    best_iter_score = []
    bar = tqdm(range(100))
    for _ in bar:
        recipes, portions = generate_new_results(recipes, portions, result_evaluations, generator, b_conf)
        result_evaluations = [fitted_nutrinator.compute(A, P) for A, P in zip(recipes, portions)]
        best_iter_score.append((min(result_evaluations)))
        bar.set_description(f"Best this iteration: {best_iter_score[-1]}")
    return best_iter_score


if __name__ == "__main__":
    days = 7
    dishes_per_day = 3

    nutrients_of_recipes = genfromtxt("data/csv/nutrients.csv", delimiter=",")
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

    story = bees_algorithm(generator, nutrinator)

    plt.plot(story)
    plt.show()
