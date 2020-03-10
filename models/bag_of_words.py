"""
Word2Vec model for recipe analysis
"""
import os
import csv
import random
import re
from collections import defaultdict

import numpy as np
from gensim.models import Word2Vec, KeyedVectors

from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from utils.exceptions import SetupError
from utils.helpers import get_gen_at_index


class RecipeAnalyzer(object):
    """
    Class to inspect and analyze the recipe data
    """

    def __init__(self, corpus, num_features=100):
        self.corpus = corpus
        self._recipe_vectors = {}
        self._ingredient_vectors = {}
        self.num_features = num_features
        self._model = None
        self._k_means = None
        self._vectors = None

    def init(self, w2v_model_path, kv_file_path):
        self.get_or_create_model(w2v_model_path)
        self.save_template(w2v_model_path)
        self.load_or_train(kv_file_path)
        self.save_trained(kv_file_path)
        self._ingredient_vectors = self.calc_ingredient_vecs()
        self._recipe_vectors = self.calc_recipe_vectors()

    @property
    def model(self):
        return self._model

    @property
    def recipe_vectors(self):
        return self._recipe_vectors

    @property
    def ingredient_vectors(self):
        return self._ingredient_vectors

    def get_or_create_model(self, path=None):
        """
        Attempt to load an existing model from path, else create one
        """
        if not path:
            path = os.path.join(os.pardir, 'created_models', 'w2v_ingredients.model')
        if not os.path.exists(path):
            self._model = Word2Vec([x for _, x in self.corpus], size=self.num_features)
        else:
            self._model = Word2Vec.load(path)

    def save_template(self, path='w2v_ingredients.model'):
        """
        Save the Word2Vec model with a name
        """
        self._model.save(path)

    def save_trained(self, path='trained_w2v_ingredients.kv'):
        """
        Save the word vectors to a .kv file with name
        """
        if not self._vectors:
            raise SetupError('Model has not yet been trained')
        self._vectors.save(path)

    def load_or_train(self, path='trained_w2v_ingredients.kv'):
        """
        Attempt to load the specified .kv file, else train model and create one
        """
        if not os.path.exists(path):
            self.train()
        else:
            self._vectors = KeyedVectors.load(path, mmap='r')
            self._recipe_vectors = self.calc_recipe_vectors()

    def train(self, epochs=5, **kwargs):
        """
        Train Word2Vec model, to generate word vectors
        """
        self._model.train(
            [item[1] for item in self.corpus], epochs=epochs, total_examples=self._model.corpus_count, **kwargs
        )
        self._vectors = self._model.wv
        self._recipe_vectors = self.calc_recipe_vectors()

    def get_word_vectors(self, words, safe=False):
        vecs = []
        for word in words:
            if not word:
                continue
            try:
                vec = self._vectors[word]
            except KeyError:
                if not safe:
                    raise
                vec = np.zeros(self.num_features)
            vecs.append(vec)
        return vecs

    def calc_recipe_vectors(self):
        """
        Calculate vectors for each recipe, by summing the individual word vectors for each ingredient
        """
        vector_dict = {}
        for idx, (name, processed_ingredients) in enumerate(self.corpus):
            vecs = []
            for ingredient_item in processed_ingredients:
                ingredient_vecs = self.get_word_vectors(ingredient_item.split(' '), safe=True)
                vecs.extend(ingredient_vecs)
            vec_sum = sum(vecs)
            if not isinstance(vec_sum, np.ndarray) or len(vec_sum) != self.num_features:
                vec_sum = np.zeros(self.num_features)
            vector_dict[name] = vec_sum
        return vector_dict

    def get_unique_ingredients(self):
        unique_ingredients = set()
        for _, ingredients in self.corpus:
            for ingredient in ingredients:
                unique_ingredients.add(ingredient)
        unique_ingredients = list(unique_ingredients)
        print(f'corpus contains {len(unique_ingredients)} unique words')
        return unique_ingredients

    def calc_ingredient_vecs(self):
        """
        Calculate vectors for each ingredient item
        """
        unique_ingredients = self.get_unique_ingredients()
        vector_dict = {}
        for ingredient in unique_ingredients:
            vecs = self.get_word_vectors(ingredient.split(' '), safe=True)
            vec_sum = sum(vecs)
            if not isinstance(vec_sum, np.ndarray) or len(vec_sum) != self.num_features:
                vec_sum = np.zeros(self.num_features)
            vector_dict[ingredient] = vec_sum
        return vector_dict

    # def save_vecs(self, path=None):
    #     """
    #     Save word vectors, ingredient vectors and recipe level vectors to JSON
    #     TODO rework this, file is way too big
    #     :return:
    #     """
    #     if not path:
    #         path = os.path.join(os.pardir, 'created_models', 'all_vectors.json')
    #     ingredient_vecs = self.calc_ingredient_vecs()
    #     recipe_vecs = self.calc_recipe_vectors()
    #     word_vecs = {
    #         self._vectors.index2word[idx]: vec
    #         for idx, vec in enumerate(self._vectors.vectors)
    #     }
    #     data = {
    #         'ingredients': {key: value.tolist() for key, value in ingredient_vecs.items()},
    #         'recipes': {key: value.tolist() for key, value in recipe_vecs.items()},
    #         'words': {key: value.tolist() for key, value in word_vecs.items()},
    #     }
    #     with open(path, 'w') as f:
    #         json.dump(data, f)

    def cluster(self, n=3):
        """
        Perform K-means clustering on the recipe vector data
        """
        self._k_means = KMeans(
            n_clusters=n, random_state=0
        ).fit(np.asarray(self._recipe_vectors.values()))

    def print_clusters(self):
        """
        Print the K-means cluster data
        """
        labels = list(self._k_means.predict(np.asarray(self._recipe_vectors.values())))
        labels_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            labels_dict[label].append(get_gen_at_index(self.corpus.get_names(), idx))
        print(labels_dict)

    def get_similar_recipes(self, name, k=3):
        """
        Get the name of the k most similar recipes to the inputted name.
        Using cosine distance
        """
        similar = self.get_similar(self._recipe_vectors, name, k=k)
        similar_str = '\n- '.join(similar)
        print(f'Top {k} most similar recipes to {name}: \n- {similar_str}')
        return similar

    def replace_ingredient(self, name, k=3):
        similar = self.get_similar(self._ingredient_vectors, name, k=k)
        similar_str = '\n- '.join(similar)
        print(f'Top {k} most similar ingredients to {name}: \n- {similar_str}')
        return similar

    def get_similar(self, vectors, name, k=3):
        target_vec = vectors[name]
        reduced_names = [key for key in vectors.keys() if key.lower() != name.lower()]
        reduced_vectors = [vectors[key] for key in vectors.keys() if key.lower() != name.lower()]
        similar_indexes = self.get_closest_indexes(reduced_vectors, target_vec, k=k)
        similar = [reduced_names[idx] for idx in similar_indexes]
        return similar

    def get_closest_indexes(self, vectors, target, k=3):
        kd_tree = cKDTree(vectors)
        _, indexes = kd_tree.query(target, k=k)
        return indexes


class BaseCorpus(object):
    def __init__(self, corpus_path=None):
        self.corpus_path = corpus_path

    def __iter__(self):
        raise NotImplementedError('No defined method for iteration')


class BaseRecipeCorpus(BaseCorpus):
    """
    Iterator that yields recipe data
    """

    def __init__(self, corpus_path=None):
        if not corpus_path:
            corpus_path = os.path.join(os.pardir, 'data', 'recipe_details.csv')
        super().__init__(corpus_path=corpus_path)
        self.column_names = []

    def __iter__(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            self.column_names = next(csv_reader)
            for line in csv_reader:
                yield tuple(line)

    @staticmethod
    def remove_unwanted_chars(item):
        """
        Remove quantities from ingredient items.
        e.g:
            - '100g/3½oz carrot, grated, peeled' -> 'carrot grated peeled'
            - '600g/1lb 5oz fresh apricots' -> 'fresh apricots'
        :param item:
        :return:
        """
        char_sets = [
            r'\d+[a-zA-Z/¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]+',
            r'\d+',
            r' x ',
            r'[().,!?\'"\-+]',
            r'[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]',
            r' tbsp|tsp|oz ',
        ]
        filtered = re.sub(r'|'.join(char_sets), '', item).strip()
        return re.sub(r' +', ' ', filtered)

    # def get_row(self, index):
    #     with open(self.corpus_path, 'r', encoding='utf-8') as f:
    #         csv_reader = csv.reader(f)
    #         return list(csv_reader)[index+1]
    #
    # # All pre-processing should be done in the model
    # def get_col_values(self, col_indexes):
    #     for line in self.__iter__():
    #         yield [line[col_index] for col_index in col_indexes]
    #
    # def get_col_value(self, col_index, index):
    #     return self.get_row(index)[col_index]
    #
    # def get_ingredient_tuples(self):
    #     for name, ingredients in self.get_col_values([1, 5]):
    #         formatted_ingredient_items = [self.pre_process_ingredient(ingredient)
    #     #                                       for ingredient in ingredients.split('::')]
    #         yield name, utils.simple_preprocess(' '.join(formatted_ingredient_items))
    #
    # def get_processed_ingredients(self):
    #     for ingredients in self.get_col_values([5]):
    #         formatted_ingredient_items = [self.pre_process_ingredient(ingredient)
    #                                       for ingredient in ingredients[0].split('::')]
    #         yield utils.simple_preprocess(' '.join(formatted_ingredient_items))
    #
    # def get_names(self):
    #     return (x[0] for x in self.get_col_values([1]))
    #
    # def get_flat_ingredients(self):
    #     for line in csv.reader(open(self.corpus_path, 'r', encoding='utf-8'), delimiter=','):
    #         if not line[0]:
    #             continue  # skip header row
    #         for ingredient in line[5].split('::'):
    #             yield self.pre_process_ingredient(ingredient)
    #
    # @staticmethod
    # def pre_process_ingredient(item):
    #     """
    #     Remove quantities from ingredient items.
    #     e.g:
    #         - '100g/3½oz carrot, grated, peeled' -> 'carrot grated peeled'
    #         - '600g/1lb 5oz fresh apricots' -> 'fresh apricots'
    #     :param item:
    #     :return:
    #     """
    #     char_sets = [
    #         r'\d+[a-zA-Z/¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]+',
    #         r'\d+',
    #         r'[().,!?\'"\-+]',
    #         r'[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]',
    #     ]
    #     return re.sub(r'|'.join(char_sets), '', item).strip()


class IngredientOnlyCorpus(BaseRecipeCorpus):
    def __init__(self, corpus_path=None):
        super().__init__(corpus_path=corpus_path)

    def __iter__(self):
        """
        Select only name and ingredients list from corpus
        :return:
        """
        for row_tuple in super(IngredientOnlyCorpus, self).__iter__():
            ingredients = row_tuple[4]
            ingredients = self.pre_process_ingredients(ingredients)
            yield row_tuple[0], ingredients

    def pre_process_ingredients(self, ingredients):
        return [self.remove_unwanted_chars(ingredient) for ingredient in ingredients.split('::')]


class TitleOnlyCorpus(BaseRecipeCorpus):
    def __init__(self, corpus_path=None):
        super().__init__(corpus_path=corpus_path)

    def __iter__(self):
        """
        Select only name and ingredients list from corpus
        :return:
        """
        for row_tuple in super(TitleOnlyCorpus, self).__iter__():
            full_name = row_tuple[1]
            processed_name = self.pre_process_name(full_name)
            yield full_name, processed_name

    def pre_process_name(self, name):
        return self.remove_unwanted_chars(name)


if __name__ == "__main__":
    _model = RecipeAnalyzer(corpus=IngredientOnlyCorpus())
    _model.init()
    for i in range(5):
        _model.get_similar_recipes(random.choice(list(_model._recipe_vectors.keys())))
# TODO:
# model loads generic corpus:
# should have one parent corpus with __iter__ that reads entire line,
# then some subclasses which only return name and ingredients
# for each item in corpus
