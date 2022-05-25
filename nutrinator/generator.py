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
    max_occurence: int = 2
    recipes_size: int = 100
    taboo: Set[int] = field(default_factory=default_taboo)
    portion_set: List[float] = field(default_factory=default_portions)


class Generator:
    def __init__(self, config: GeneratorConfig = GeneratorConfig()):
        self.config = config
        self.shape = (self.config.dishes,)
        # self.shape = (self.config.days, self.config.dishes)
        self.recipes = {i for i in range(config.recipes_size)} - config.taboo
        self.generator = np.random.default_rng()

    def generate_days(self, size: int = None, previous: np.ndarray = np.array([])) -> Tuple[np.ndarray, np.ndarray]:
        size = size if size else self.config.days  # przepraszam, że to tak brzydko wygląda :c
        used_recipes, used_count = np.unique(previous, return_counts=True)
        overused_recipes = set(used_recipes[used_count > self.config.max_occurence])
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

    def generate_neighbours(self, best_results: List, n: int):
        return [self.generate_days(self.config.days) for _ in range(n)]