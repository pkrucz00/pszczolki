import requests
import numpy as np
from time import sleep

URL = "https://api.edamam.com/api/recipes/v2"
APP_ID = "7ede0f0e"
APP_KEY = "56203261276e6c85344964c122ad3e5d"

NUTRIENT_JSON_NAMES = ["ENERC_KCAL", "FAT", "CHOCDF", "PROCNT"]

OUTPUT_NUTRIENT_FILE = "csv/nutrients2.csv"
OUTPUT_INGREDIENT_FILE = "csv/ingredients2.csv"


def get_nutrients_and_ingredients(search_string):
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "type": "public",
        "q": search_string
    }
    r = requests.get(URL, params=params)
    response_dict = r.json()
    result_nutrients, result_ingredients = [], []
    hits = response_dict["hits"]
    for hit in hits:
        recipe = hit["recipe"]
        nutrients = recipe["totalNutrients"]
        nutrient_table = [nutrients[nutrient_json_name]["quantity"] for nutrient_json_name in NUTRIENT_JSON_NAMES]
        ingredient_table = {ingredient["food"]: ingredient["quantity"] for ingredient in recipe["ingredients"] if ingredient["quantity"] > 0}
        result_nutrients.append(nutrient_table)
        result_ingredients.append(ingredient_table)

    return result_nutrients, result_ingredients


def nutrient_list_to_np_array(nutrient_list):
    return np.array(nutrient_list)


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


if __name__ == '__main__':
    with open("food.txt") as file:
        foods_list = [name.strip() for name in file.readlines()]

    nutrient_list, ingredient_list = [], []

    for food_name in foods_list:
        print(f"Downloading {food_name} info...")
        nutrients, ingredients = get_nutrients_and_ingredients(food_name)
        print(f"Nutrients and ingredients for {food_name} successfully downloaded!")
        print(f"Number of recipes analyzed: {len(nutrients)}")
        print()

        nutrient_list.extend(nutrients)
        ingredient_list.extend(ingredients)
        sleep(6.4)   # ograniczenia związane z API (maks. 10 zapytań na minutę)

    print(f"Download completed! Total recipes analyzed: {len(nutrient_list)}")
    ingredients_index = create_ingredient_index(ingredient_list)
    print(f"Total number of different ingredients: {len(ingredients_index)}")

    nutrient_array = nutrient_list_to_np_array(nutrient_list)
    ingredient_array = ingredients_to_array(ingredient_list, ingredients_index)

    np.savetxt(OUTPUT_NUTRIENT_FILE, nutrient_array, delimiter=",")
    print(f"Nutrients saved at location {OUTPUT_NUTRIENT_FILE}")

    ingredient_array.tofile(OUTPUT_INGREDIENT_FILE, delimiter=",")
    print(f"Ingredients saved at location {OUTPUT_INGREDIENT_FILE}")
