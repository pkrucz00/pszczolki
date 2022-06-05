import numpy as np
import json
from tqdm import tqdm
from numpy import genfromtxt
from uuid import uuid4

from dataclasses import dataclass
from typing import List, Tuple, Optional

from nutrinator.generator import Generator
from nutrinator.generator import GeneratorConfig as GenConf
from nutrinator.nutrionator import Nutrinator


@dataclass
class BeesConfig:
    n: int = 100
    m: int = 30
    e: int = 10
    nsp: int = 50
    nep: int = 100
    max_iter: int = 10
    max_stagnation: int = 10
    include_original: bool = False
    use_best_of_all: bool = False


def stop_criterion(i):
    return i < 10000


class Solver:
    _MACRO_SIZE = 4

    def __init__(self, generator_config: GenConf, config: BeesConfig, file: str = "csv/nutrients.csv"):
        self._bee_config = config
        self._gen_config = generator_config
        nutrients_of_recipes = genfromtxt(file, delimiter=",")
        self.nut = Nutrinator(nutrients_of_recipes)
        self.generator = Generator(self._gen_config)
        self._fitted = False

    def fit(self, demand: np.ndarray, special_demand: Optional[np.ndarray] = None):
        np_demand = np.array([demand for _ in range(self._gen_config.days)], dtype=float)
        if special_demand is None:
            special_demand = np.zeros((*self._shape, self._MACRO_SIZE), dtype=float)
        self.nut.fit(np_demand, special_demand, self._generate_gamma(demand))
        self._fitted = True

    @staticmethod
    def _generate_gamma(demand):
        return np.array([(1/x) for x in demand], dtype=float)

    def run(self, special_settings: Optional[BeesConfig] = None, verbose: bool = False):
        if special_settings is None:
            special_settings = self._bee_config
        if not self._fitted:
            raise RuntimeError('Solver was not fitted')
        return self._bees_algorithm(special_settings, verbose)

    def _bees_algorithm(self, b_conf: BeesConfig, verbose: bool):
        recipes, portions = self.generator.generate_days(b_conf.n)
        result_evaluations = [self.nut.compute(A, P) for A, P in zip(recipes, portions)]
        best_iter_score = []
        if verbose:
            bar = tqdm(range(b_conf.max_iter))
        else:
            bar = range(b_conf.max_iter)

        stagnation = 0
        current_best_score = 99999999999999999999999
        best_recipes, best_portions = None, None
        big_table = {}

        for i in bar:
            recipes, portions = self._generate_iteration(recipes, portions, result_evaluations, b_conf)
            result_evaluations = [self.nut.compute(A, P) for A, P in zip(recipes, portions)]

            big_table[i] = result_evaluations.copy()

            best_iter = min(result_evaluations)

            best_iter_score.append(best_iter)
            if best_iter < current_best_score:
                index = result_evaluations.index(best_iter)
                best_recipes = recipes[index, ...]
                best_portions = portions[index, ...]
                current_best_score = best_iter
                stagnation = 0
            else:
                stagnation += 1

            if b_conf.max_stagnation <= stagnation:
                if verbose:
                    print(f'Result stagnated for {stagnation}')
                break
            if verbose:
                bar.set_description(f"Best this iteration: {best_iter_score[-1]}, current best {current_best_score}")
        return best_iter_score, best_recipes, best_portions, big_table

    @staticmethod
    def save_output(raw_data: dict, filename) -> None:
        with open(filename, 'w') as f:
            json.dump(raw_data, f, sort_keys=True, indent=4, separators=(',', ': '))

    @staticmethod
    def verbose_output(_story, recipes, portions, raw_data):
        print('Recipes')
        print(recipes)
        print('')
        print('Portions')
        print(portions)
        file = f'{uuid4()}.json'
        with open(file, 'w') as f:
            json.dump(raw_data, f, sort_keys=True, indent=4, separators=(',', ': '))
        print(f'Raw data saved in {file}')

    @property
    def _shape(self):
        return self._gen_config.days, self._gen_config.dishes

    def _explore_nhbd(self, nhbd_size: int,
                      recipes: np.ndarray, portions: np.ndarray,
                      include_original: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        original_size = recipes.shape[0]
        nhbd_recipes = np.empty((original_size, *self._shape), dtype=int)
        nhbd_portions = np.empty((original_size, *self._shape), dtype=float)

        for idx, (recipe, portion) in enumerate(zip(recipes, portions)):
            if include_original:
                nhbd_recipes[idx, ...] = recipe
                nhbd_portions[idx, ...] = portion
                best_fit = self.nut.compute(recipe, portion)
            else:
                best_fit = 999999999999999999999

            for _ in range(nhbd_size):
                # ma zwracac tupla
                recruited_bee = self.generator.generate_neighbour(recipe, portion)
                fit = self.nut.compute(recruited_bee[0], recruited_bee[1])
                if fit < best_fit:
                    nhbd_recipes[idx, ...] = recruited_bee[0]
                    nhbd_portions[idx, ...] = recruited_bee[1]
                    best_fit = fit

        return nhbd_recipes, nhbd_portions

    def _generate_iteration(self, recipes: np.ndarray, portions: np.ndarray,
                            results: List[float], conf: BeesConfig):
        indices = np.argsort(results)
        result_recipes, result_portions = recipes[indices], portions[indices]

        e_result_recipes, e_result_portions = result_recipes[:conf.e, ...], result_portions[:conf.e, ...]
        m_result_recipes, m_result_portions = result_recipes[conf.e:conf.m, ...], result_portions[conf.e:conf.m, ...]

        nep_nhbd_results = self._explore_nhbd(conf.nep, e_result_recipes, e_result_portions, conf.include_original)
        nsp_nhbd_results = self._explore_nhbd(conf.nsp, m_result_recipes, m_result_portions, conf.include_original)
        random_results = self.generator.generate_days(conf.n - conf.m)

        if conf.use_best_of_all:
            nep_nhbd_results = self._stack_best(*nep_nhbd_results, e_result_recipes, e_result_portions, conf.e)
            nsp_nhbd_results = self._stack_best(*nsp_nhbd_results, m_result_recipes, m_result_portions, conf.m - conf.e)

        best_recipes = np.concatenate((nep_nhbd_results[0], nsp_nhbd_results[0], random_results[0]), axis=0)
        best_portions = np.concatenate((nep_nhbd_results[1], nsp_nhbd_results[1], random_results[1]), axis=0)

        return best_recipes, best_portions

    def _stack_best(self, nhbd_recipes, nhbd_portions, old_recipes, old_portions, size):
        recipes = np.concatenate((nhbd_recipes, old_recipes), axis=0)
        portions = np.concatenate((nhbd_portions, old_portions), axis=0)
        scores = [self.nut.compute(recipe, portion) for recipe, portion in zip(recipes, portions)]
        indices = np.argsort(scores)
        return recipes[indices[:size]], portions[indices[:size]]


if __name__ == '__main__':
    bee_config = BeesConfig(
        max_iter=50000,
        max_stagnation=3000,
        include_original=True
    )

    gen_config = GenConf()

    demand = [2456, 70, 327, 150]

    solver = Solver(gen_config, bee_config)

    solver.fit(demand)

    output = solver.run(verbose=True)

#
# def gather_data(bees_set: List[BeesConfig]):
#     days = 7
#     dishes_per_day = 3
#
#
#
#     #               kcal, fat, fat, protein
#     needed_macro = [2456, 70, 327, 150]
#     # gamma should make every part equal in score
#     gamma = np.array([(1 / 2456), (1 / 70), (1 / 327), (1 / 150)], dtype=float)
#
#     nutrinator = Nutrinator(nutrients_of_recipes, gamma=gamma)
#
#     nutrient_demand = np.array([needed_macro for _ in range(7)], dtype=float)
#     nutrient_special_demand = np.zeros((days, dishes_per_day, 4))
#
#     nutrinator.fit(nutrient_demand, nutrient_special_demand)
#
#     for config in bees_set:
#         generator = Generator(GenConf(days=days,
#                                       dishes=dishes_per_day,
#                                       recipes_size=nutrients_of_recipes.shape[0]))
#
#         data = bees_algorithm(generator, nutrinator)
#         verbose_output(config, *data)
