import random

import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple


def default_portions() -> List[float]:
    return [0.25, 0.5, 1, 1.5, 2]


def default_taboo() -> Set[int]:
    return set()


@dataclass
class GeneratorConfig:
    # If we do the greedy way, we won't need days
    days: int = 7
    dishes: int = 5
    max_occurrence: int = 2
    recipes_size: int = 100
    taboo: Set[int] = field(default_factory=default_taboo)
    portion_set: List[float] = field(default_factory=default_portions)


class Generator:
    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        self.config = config
        self.shape = (self.config.days, self.config.dishes)
        self.recipes = {i for i in range(config.recipes_size)} - config.taboo
        self.generator = np.random.default_rng()

    def generate_days(self, size: int, previous: np.ndarray = np.array([])) -> Tuple[np.ndarray, np.ndarray]:
        used_recipes, used_count = np.unique(previous, return_counts=True)
        overused_recipes = set(used_recipes[used_count > self.config.max_occurrence])
        current_recipes = list(self.recipes - overused_recipes)
        recipes = np.empty((size, *self.shape), dtype=int)
        # don't know how to do it fully in numpy, if we generate all days at once we can:
        #   - have not enough recipes with replace=False
        #   - have same dishes in the same day with replace=True, and I think validating that will be more costly
        for sample in range(size):
            recipes[sample, ...] = self.__generate_recipes(current_recipes)

        return recipes, self.__generate_portions(size)

    def __generate_recipes(self, allowed: List[int]) -> np.ndarray:
        return self.generator.choice(allowed, self.shape, replace=False)

    def __generate_portions(self, size: int) -> np.ndarray:
        return self.generator.choice(self.config.portion_set, (size, *self.shape), replace=True)

    def generate_neighbour(self, recipes: np.ndarray, portions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # do note that we change recipes and portions in place
        recipes = np.copy(recipes)
        portions = np.copy(portions)

        neighbour = self.__generate_neighbour(recipes, portions)
        while not self.__check_constraints(neighbour[0]):
            neighbour = self.__generate_neighbour(recipes, portions)

        return neighbour

    # def generate_neighbours(self, recipes: np.ndarray, portions: np.ndarray, n: int):
    #     # return self.generate_days(n)  #TODO change it to a real generator
    #     print(recipes.shape)
    #     new_recipes = np.empty((n*recipes.shape[0], *self.shape), dtype=int)
    #     new_portions = np.empty((n*recipes.shape[0], *self.shape), dtype=float)
    #     for i in range(recipes.shape[0]):
    #         for j in range(n):
    #             neighbour = self.__generate_neighbour(recipes[i], portions[i])
    #             while not self.__check_constraints(neighbour[0]):
    #                 neighbour = self.__generate_neighbour(recipes[i], portions[i])
    #             new_recipes[i*n+j, ...] = neighbour[0]
    #             new_portions[i*n+j, ...] = neighbour[1]
    #
    #     return new_recipes, new_portions
    def __generate_day_and_dish(self):
        day = self.generator.integers(0, self.config.days)
        dish = self.generator.integers(0, self.config.dishes)
        return day, dish

    def __generate_nhbd_change_dish(self, recipes: np.ndarray, portions: np.ndarray):
        day, dish = self.__generate_day_and_dish()
        old_recipe = recipes[day][dish]
        recipes[day][dish] = self.generator.choice(list(set(self.recipes) - {old_recipe}))
        portions[day][dish] = 1
        return recipes, portions

    def __generate_nhbd_change_portion(self, recipes: np.ndarray, portions: np.ndarray):
        day, dish = self.__generate_day_and_dish()
        old_portion = portions[day][dish]
        portions[day][dish] = self.generator.choice(list(set(self.config.portion_set) - {old_portion}))
        return recipes, portions

    def __generate_nhbd_swap_dishes(self, recipes: np.ndarray, portions: np.ndarray):
        day1, dish1 = self.__generate_day_and_dish()
        day2, dish2 = self.__generate_day_and_dish()
        while day1 == day2 and dish1 == dish2:
            day2, dish2 = self.__generate_day_and_dish()

        buf_recipe, buf_portion = recipes[day1, dish1], portions[day1, dish1]
        recipes[day1, dish1], portions[day1, dish1] = recipes[day2, dish2], portions[day2, dish2]
        recipes[day2, dish2], portions[day2, dish2] = buf_recipe, buf_portion
        return recipes, portions

    def __generate_neighbour(self, recipes: np.ndarray, portions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nhbd = random.choice(['SWAP', 'PORTION', 'RECIPE'])

        if nhbd == 'SWAP':
            return self.__generate_nhbd_swap_dishes(recipes, portions)
        if nhbd == 'PORTION':
            return self.__generate_nhbd_change_portion(recipes, portions)
        if nhbd == 'RECIPE':
            return self.__generate_nhbd_change_dish(recipes, portions)

    def __create_dict(self, recipes: np.ndarray) -> dict:
        data = dict()
        for i in range(self.config.days):
            for j in range(self.config.dishes):
                if recipes[i][j] in data.keys():
                    data[recipes[i][j]].append(i)
                else:
                    data[recipes[i][j]] = [i]
        return data

    # if something's wrong, return False
    def __check_constraints(self, recipes: np.ndarray) -> bool:
        data = self.__create_dict(recipes)

        for key, values in list(data.items()):
            if len(values) > 3:
                # print(key, ": too many occurrences")
                return False

            for i in range(1, len(values)):
                if values[i] == values[i - 1]:
                    # print(key, ": same day")
                    return False
                if (values[i] - values[i - 1]) < 2:
                    # print(key, ": small break")
                    return False

        return True
