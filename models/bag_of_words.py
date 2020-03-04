"""
Word2Vec model for recipe analysis
"""
import json
import os
import csv
import re
from collections import defaultdict

import numpy as np
from gensim import utils
from gensim.models import Word2Vec, KeyedVectors

from sklearn.cluster import KMeans
from scipy.spatial import KDTree

from recipes.utils.helpers import get_gen_at_index


class RecipeAnalyzer(object):
    """
    Class to inspect and analyze the recipe data
    """

    def __init__(self):
        self.corpus = RecipeCorpus()
        self._recipe_vectors = {}
        self._ingredient_vectors = {}
        self._model = None
        self._k_means = None
        self._vectors = None

    def init(self):
        pass

    @property
    def model(self):
        return self._model

    def get_or_create_model(self, path=None):
        """
        Attempt to load an existing model from path, else create one
        """
        if not path:
            path = os.path.join('.', 'created_models', 'w2v_ingredients.model')
        if not os.path.exists(path):
            self._model = Word2Vec([x for x in self.corpus.get_processed_ingredients()])
        else:
            self._model = Word2Vec.load(path)

    def save_template(self, name='w2v_ingredients.model'):
        """
        Save the Word2Vec model with a name
        """
        self._model.save(os.path.join('.', 'created_models', name))

    def save_trained(self, name='trained_w2v_ingredients.kv'):
        """
        Save the word vectors to a .kv file with name
        """
        if not self._vectors:
            raise SetupError('Model not yet trained')
        self._vectors.save(os.path.join('.', 'created_models', name))

    def load_or_train(self, name='trained_w2v_ingredients.kv'):
        """
        Attempt to load the specified .kv file, else train model and create one
        """
        path = os.path.join('.', 'created_models', name)
        if not os.path.exists(path):
            self.train()
        else:
            self._vectors = KeyedVectors.load(path, mmap='r')
            self._recipe_vectors = self.calc_recipe_vectors()

    def train(self, epochs=5, **kwargs):
        """
        Train Word2Vec model, to generate word vectors
        """
        self._model.train(self.corpus.get_processed_ingredients(), epochs=epochs,
                          total_examples=self._model.corpus_count, **kwargs)
        self._vectors = self._model.wv
        self._recipe_vectors = self.calc_recipe_vectors()

    def calc_recipe_vectors(self):
        """
        Calculate vectors for each recipe, by summing the individual word vectors for each ingredient
        """
        vector_dict = {}
        for idx, (name, processed_ingredients) in enumerate(self.corpus.get_ingredient_tuples()):
            vecs = []
            for word in processed_ingredients:
                try:
                    vec = self._vectors[word]
                except KeyError:
                    vec = np.zeros(100)
                vecs.append(vec)
            vec_sum = sum(vecs)
            if not isinstance(vec_sum, np.ndarray) or len(vec_sum) != 100:
                vec_sum = np.zeros(100)
            vector_dict[name] = vec_sum
        return vector_dict

    def calc_ingredient_vecs(self):
        """
        Calculate vectors for each ingredient item
        """
        ingredients = self.corpus.get_flat_ingredients()
        unique_ingredients = set(list(ingredients))
        vector_dict = {}
        for ingredient in unique_ingredients:
            if not ingredient:
                continue
            vecs = []
            for word in ingredient.split(' '):
                try:
                    vec = self._vectors[word]
                except KeyError:
                    vec = np.zeros(100)
                vecs.append(vec)
            vec_sum = sum(vecs)
            if not isinstance(vec_sum, np.ndarray) or len(vec_sum) != 100:
                vec_sum = np.zeros(100)
            vector_dict[ingredient] = vec_sum
        return vector_dict

    def save_vecs(self, path=None):
        """
        Save word vectors, ingredient vectors and recipe level vectors to JSON
        :return:
        """
        if not path:
            path = os.path.join('.', 'created_models', 'all_vectors.json')
        ingredient_vecs = self.calc_ingredient_vecs()
        recipe_vecs = self.calc_recipe_vectors()
        word_vecs = {
            self._vectors.index2word[idx]: vec
            for idx, vec in enumerate(self._vectors.vectors)
        }
        data = {
            'ingredients': {key: value.tolist() for key, value in ingredient_vecs.items()},
            'recipes': {key: value.tolist() for key, value in recipe_vecs.items()},
            'words': {key: value.tolist() for key, value in word_vecs.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_vecs(self, path=None):
        if not path:
            os.path.join('.', 'created_models', 'all_vectors.json')
        data = json.load(path)
        self._recipe_vectors = data['recipes']
        self._ingredient_vectors = data['ingredients']
        self._vectors = data['words']

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

    def get_similar_to(self, recipe_name, k=5):
        """
        Get the name of the k most similar recipes to the inputted name.
        Using cosine distance
        """
        target_vec = self._recipe_vectors[recipe_name]
        reduced_rec_vectors = {
            key: self._recipe_vectors[key] for key in self._recipe_vectors.keys() if key != recipe_name
        }
        kd_tree = KDTree(np.asarray(reduced_rec_vectors.values()))
        _, indexes = kd_tree.query(target_vec, k=k)
        similar = [get_gen_at_index(self.corpus.get_names(), idx) for idx in indexes]
        similar_str = '\n- '.join(similar)
        print(f'Top {k} most similar recipes to {recipe_name}: \n- {similar_str}')
        return similar

    def replace_ingredient(self, missing_ingredient, k=5):
        """
        Find the k most similar ingredients to a specified ingredient name.
        Uses cosine distance
        """
        replacements = self.model.wv.most_similar(positive=[missing_ingredient], topn=k)
        print(f'Replacing {missing_ingredient} with: \n{replacements}')
        return replacements


class RecipeCorpus(object):
    """
    Iterator that yields recipe data
    """

    def __init__(self, corpus_path=None):
        if not corpus_path:
            corpus_path = os.path.join('.', 'data', 'bbc', 'recipe_details.csv')
        self.corpus_path = corpus_path
        self._names = []
        self.column_names = []

    def __iter__(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            self.column_names = next(csv_reader)
            for line in csv_reader:
                yield line

    def get_row(self, index):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            return list(csv_reader)[index+1]

    def get_col_values(self, col_indexes):
        for line in self.__iter__():
            yield [line[col_index] for col_index in col_indexes]

    def get_col_value(self, col_index, index):
        return self.get_row(index)[col_index]

    def get_ingredient_tuples(self):
        for name, ingredients in self.get_col_values([1, 5]):
            formatted_ingredient_items = [self.pre_process_ingredient(ingredient)
                                          for ingredient in ingredients.split('::')]
            yield name, utils.simple_preprocess(' '.join(formatted_ingredient_items))

    def get_processed_ingredients(self):
        for ingredients in self.get_col_values([5]):
            formatted_ingredient_items = [self.pre_process_ingredient(ingredient)
                                          for ingredient in ingredients[0].split('::')]
            yield utils.simple_preprocess(' '.join(formatted_ingredient_items))

    def get_names(self):
        return (x[0] for x in self.get_col_values([1]))

    def get_name(self, index):
        return self.get_col_value(1, index)

    def get_flat_ingredients(self):
        for line in csv.reader(open(self.corpus_path, 'r', encoding='utf-8'), delimiter=','):
            if not line[0]:
                continue  # skip header row
            for ingredient in line[5].split('::'):
                yield self.pre_process_ingredient(ingredient)

    @staticmethod
    def pre_process_ingredient(item):
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
            r'[().,!?\'"\-+]',
            r'[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]',
        ]
        return re.sub(r'|'.join(char_sets), '', item).strip()


class SetupError(BaseException):
    pass


if __name__ == "__main__":
    _model = RecipeAnalyzer()
    _model.get_or_create_model()
    _model.save_template()
    _model.load_or_train()
    _model.save_trained()
