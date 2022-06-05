import time

import requests
import numpy as np
import json

from time import sleep
from dataclasses import dataclass


@dataclass
class StaticKeys:
    ingredient: str = "nutrients"
    recipe: str = "recipes"
    nutrient: str = "nutrient"


@dataclass
class DownloadConfig:
    url = "https://api.edamam.com/api/recipes/v2"
    app_id = "7ede0f0e"
    app_key = "56203261276e6c85344964c122ad3e5d"

    input_food_file = "food.txt"

    nutrient_json_names = ["ENERC_KCAL", "FAT", "CHOCDF", "PROCNT"]
    min_calories = 300    # PER SERVING
    max_calories = 1000
    recipes_per_food_name = 60  # rounded up to the nearest multiple of 20 because there are 20 recipes per page


KEYS = StaticKeys()
CONFIG = DownloadConfig()

OUTPUT_MATRICES = {KEYS.nutrient: "data/csv/nutrients.csv",
                   KEYS.ingredient: "data/csv/ingredients.csv"}
OUTPUT_DICTS = {KEYS.ingredient: "data/json/ingredient_names.json",
                KEYS.recipe: "data/json/recipe_names.json",
                KEYS.nutrient: "data/json/nutrient_names.json"}


def get_next_link(links):
    next_link = links.get("next", None)
    return next_link if next_link else None


def get_nutrients_and_ingredients(search_string, url):
    params = {
        "app_id": CONFIG.app_id,
        "app_key": CONFIG.app_key,
        "calories": f"{CONFIG.min_calories}-{CONFIG.max_calories}",   # PER SERVING
        "type": "public",
        "q": search_string
    }
    r = requests.get(url, params=params)
    response_dict = r.json()
    result_nutrients, result_ingredients, recipe_names = [], [], []
    hits = response_dict["hits"]

    for hit in hits:
        recipe = hit["recipe"]
        recipe_names.append(recipe["label"])

        nutrients = recipe["totalNutrients"]
        servings = recipe["yield"]
        nutrient_table = get_nutrient_table(nutrients, servings)
        ingredient_table = {ingredient["food"]: (ingredient["quantity"] / servings)
                            for ingredient in recipe["ingredients"] if ingredient["quantity"] > 0}

        result_nutrients.append(nutrient_table)
        result_ingredients.append(ingredient_table)

    return result_nutrients,\
           result_ingredients,\
           recipe_names,\
            get_next_link(response_dict["_links"])


def get_all_nutrients_and_ingredients(search_string):
    result_nutrients, result_ingredients, recipe_names = [], [], []
    url = CONFIG.url
    while url and len(result_nutrients) < CONFIG.recipes_per_food_name:
        nutrients, ingredients, names, url = get_nutrients_and_ingredients(search_string, url)
        result_nutrients.extend(nutrients)
        result_ingredients.extend(ingredients)
        recipe_names.extend(names)
        print(f"{len(result_nutrients)}...", end=" ")
        sleep(6.4)  # ograniczenia związane z API (maks. 10 zapytań na minutę)
    return result_nutrients, result_ingredients, recipe_names


def get_nutrient_table(nutrients, servings):
    return [nutrients[nutrient_json_name]["quantity"]/servings
            for nutrient_json_name in CONFIG.nutrient_json_names]


def nutrient_list_to_np_array(nutrient_list):
    return np.array(nutrient_list)


def list_to_dict(arr):
    return {ind: val for ind, val in enumerate(arr)}


def create_ingredient_index(ingredient_list):
    return sorted(list(
            {ingredient_name
             for food_ingredients in ingredient_list
             for ingredient_name in food_ingredients.keys()}))


def ingredients_to_array(ingredient_list, ingredients_index):
    n = len(ingredients_index)
    ingredient_array = []

    for food_ingredients in ingredient_list:
        ingredient_vector = np.zeros(n)
        for ingr_name, ingr_value in food_ingredients.items():
            ind = ingredients_index.index(ingr_name)
            ingredient_vector[ind] = ingr_value
        ingredient_array.append(ingredient_vector)

    return np.array(ingredient_array)


def save_as_json(collection, dest):
    json_rep = json.dumps(collection)
    with open(dest, "w", encoding="UTF-8") as file:
        file.write(json_rep)


def save_list_to_output_file(arr, out_dir_key):
    save_as_json(list_to_dict(arr), OUTPUT_DICTS[out_dir_key])
    print(f"Nutrients names saved at {OUTPUT_DICTS[out_dir_key]}")


if __name__ == '__main__':
    with open(CONFIG.input_food_file) as file:
        foods_list = [name.strip() for name in file.readlines()]

    nutrient_list, ingredient_list, recipes_list = [], [], []
    start = time.time()
    for food_name in foods_list:
        print(f"Downloading {food_name} info...")
        nutrients, ingredients, recipes = get_all_nutrients_and_ingredients(food_name)
        print(f"Nutrients and ingredients for {food_name} successfully downloaded!")
        print(f"Number of recipes analyzed: {len(nutrients)}")
        print()

        nutrient_list.extend(nutrients)
        ingredient_list.extend(ingredients)
        recipes_list.extend(recipes)
    end = time.time()
    print(f"Total download time (with sleeps): {end-start} [s]")

    print(f"Download completed! Total recipes analyzed: {len(nutrient_list)}")
    ingredients_index = create_ingredient_index(ingredient_list)
    print(f"Total number of different ingredients: {len(ingredients_index)}")

    nutrient_array = nutrient_list_to_np_array(nutrient_list)
    ingredient_array = ingredients_to_array(ingredient_list, ingredients_index)

    np.savetxt(OUTPUT_MATRICES[KEYS.nutrient], nutrient_array, delimiter=",", fmt="%4f")
    print(f"Nutrients saved at location {OUTPUT_MATRICES[KEYS.nutrient]}")

    np.savetxt(OUTPUT_MATRICES[KEYS.ingredient], ingredient_array, delimiter=",", fmt="%1f")
    print(f"Ingredients saved at location {OUTPUT_MATRICES[KEYS.ingredient]}")

    save_list_to_output_file(recipes_list, KEYS.recipe)
    save_list_to_output_file(["Calories", "Fat", "Carbohydrates", "Protein"], KEYS.nutrient)
    save_list_to_output_file(ingredients_index, KEYS.ingredient)
