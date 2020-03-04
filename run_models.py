"""
Run the models
"""
from recipes.models.bag_of_words import RecipeAnalyzer


def run():
    rec_analyser = RecipeAnalyzer()
    rec_analyser.get_or_create_model()
    rec_analyser.save_template()
    rec_analyser.load_or_train()
    rec_analyser.save_trained()
    rec_analyser.save_vecs()
    # TODO need to split classes to have one for W2V model, then one for data analysis


if __name__ == "__main__":
    run()
