"""
Run the models
"""
import os
import random

from models.bag_of_words import RecipeAnalyzer, IngredientOnlyCorpus


def run():
    rec_analyser = RecipeAnalyzer(IngredientOnlyCorpus(corpus_path=os.path.join('data', 'recipe_details.csv')))
    rec_analyser.init('w2v_ingredients.model', 'trained_w2v_ingredients.kv')
    for i in range(5):
        rec_analyser.get_similar_to(random.choice(list(rec_analyser.recipe_vectors.keys())))
    for i in range(5):
        rec_analyser.replace_ingredient(random.choice(list(rec_analyser.ingredient_vectors.keys())))


if __name__ == "__main__":
    run()
