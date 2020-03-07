"""
Run the models
"""
from recipes.models.bag_of_words import RecipeAnalyzer, IngredientOnlyCorpus


def run():
    rec_analyser = RecipeAnalyzer(IngredientOnlyCorpus)
    rec_analyser.get_or_create_model()
    rec_analyser.save_template()
    rec_analyser.load_or_train()
    rec_analyser.save_trained()
    rec_analyser.save_vecs()


if __name__ == "__main__":
    run()
