import numpy as np
import json
import sys

from tqdm import tqdm
from numpy import genfromtxt
from uuid import uuid4

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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


class Solver:
    _MACRO_SIZE = 4

    def __init__(self, generator_config: GenConf, config: BeesConfig, file: str = "data/csv/nutrients.csv"):
        nutrients_of_recipes = genfromtxt(file, delimiter=",")
        generator_config.recipes_size = nutrients_of_recipes.shape[0]

        self.nut = Nutrinator(nutrients_of_recipes)
        self._bee_config = config
        self._gen_config = generator_config
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


def load_json(path):
    with open(path, "r", encoding="UTF-8") as file:
        return json.loads(" ".join(file.readlines()))


def load_recipe_json(path):
    return {int(recipe_ind): recipe_name
            for recipe_ind, recipe_name in load_json(path).items()}


def get_recipe_matrix(recipe_indexes):
    recipe_dict = load_recipe_json("data/json/recipe_names.json")
    return [[recipe_dict[ind] for ind in recipes] for recipes in recipe_indexes]


def print_title(number_of_days):
    title = f"YOUR FOOD PLAN for {number_of_days} days"
    print(title)
    print("="*len(title))


def print_dish(recipe, portion, indent="\t"):
    print(indent + f"Name:     {recipe}")
    print(indent + f"Portion:  {portion}")


def print_plan_for_day(day_num, recipes, portions):
    print(f"DAY {day_num}:")
    for i, recipe_info in enumerate(zip(recipes, portions)):
        recipe_num = i + 1
        print(f"Dish no {recipe_num}:")
        print_dish(*recipe_info)


def pretty_print_output(recipe_indexes, portions):
    print_title(recipe_indexes.shape[0])
    for i, (recipes, portions) in enumerate(zip(get_recipe_matrix(recipe_indexes), portions)):
        print_plan_for_day(i+1, recipes, portions)
        print()


def get_bees_config(bee_conf_dict):
    bee_conf_obj = BeesConfig()
    for attr_name, attr_value in bee_conf_dict.items():
        setattr(bee_conf_obj, attr_name, attr_value)
    return bee_conf_obj


def get_generation_input(gen_conf_dict):
    gen_conf_obj = GenConf()
    for attr_name, attr_value in gen_conf_dict.items():
        setattr(gen_conf_obj, attr_name, attr_value)
    return gen_conf_obj


def get_macros_vector(macro_info: Dict, nutri_ind_dict):
    n = len(nutri_ind_dict)
    result_vector = np.zeros(n)
    for macro_name, macro_val in macro_info.items():
        ind = nutri_ind_dict[macro_name]
        result_vector[ind] = macro_val
    return result_vector


def compute_BMR(user_data: Dict):
    sex, weight, height, age = \
        user_data["sex"], user_data["weight"], user_data["height"], user_data["age"]

    if sex not in ("MALE", "FEMALE"):
        raise Exception(f"{sex} is not sex")

    bmr_male = 88.36 + 13.4*weight + 4.8 * height - 5.68 * age
    bmr_female = 447.6 + 9.25 * weight + 3.1 * height - 4.33 * age
    return bmr_male if sex == "MALE" else bmr_female


def compute_calories(user_data: Dict):
    activity_multipliers = {"LOW": 1.2, "MODERATE": 1.5, "HIGH": 1.8}
    activity = user_data["activity"]

    if activity not in activity_multipliers:
        raise Exception(f"{activity} is not in allowed activities")

    return compute_BMR(user_data) * activity_multipliers[activity]


def compute_demand(user_data: Dict):
    def get_macro_formula(percent_of_cals, cals_per_gram):
        return lambda total_cals: percent_of_cals * total_cals / cals_per_gram

    carbs_func = get_macro_formula(0.5, 4)
    fat_func = get_macro_formula(0.25, 9)
    protein_func = get_macro_formula(0.25, 4)

    calories = compute_calories(user_data)
    result = [calories, fat_func(calories), carbs_func(calories), protein_func(calories)]
    return [round(x) for x in result]

def get_special_demand(special_demand_list, dimensions, nutri_ind_path="data/json/nutrient_indecies.json"):
    nutri_ind_dict = load_json(nutri_ind_path)
    no_nutrients = len(nutri_ind_dict)
    result = np.zeros((*dimensions, no_nutrients))
    for demand_info in special_demand_list:
        i, j = demand_info["day_number"] - 1, demand_info["dish_number"] - 1
        for macro_name, macro_val in demand_info["macros"].items():
            k = nutri_ind_dict[macro_name]
            result[i, j, k] = macro_val
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: bees_algorithm.py path/to/input.json")
        exit(1)

    input_dict = load_json(sys.argv[1])
    bee_config = get_bees_config(input_dict["🐝Config"])
    gen_config = GenConf()

    demand = compute_demand(input_dict["user_data"])
    special_demand = get_special_demand(input_dict.get("special_demand", []), (gen_config.days, gen_config.dishes))

    solver = Solver(gen_config, bee_config)
    solver.fit(demand)

    _, best_recipes, best_portions, _ = solver.run(verbose=True)
    pretty_print_output(best_recipes, best_portions)