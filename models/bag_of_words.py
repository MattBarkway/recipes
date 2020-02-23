import os
import csv
import re
from collections import defaultdict

import numpy as np
from gensim import utils
from gensim.models import Word2Vec, KeyedVectors

from sklearn.cluster import KMeans
from scipy.spatial import KDTree


class RecipeCorpus(object):
    """
    Iterator that yields lists of ingredients
    """

    def __init__(self, corpus_path=None):
        if not corpus_path:
            corpus_path = os.path.join('..', 'data', 'bbc', 'recipe_details.csv')
        self.corpus_path = corpus_path
        self._names = []
        self.rec_vecs = None
        self._model = None
        self.k_means = None
        self.vectors = None

    @property
    def model(self):
        return self._model

    def __iter__(self):
        for ingredients in self.get_ingredients():
            yield utils.simple_preprocess(' '.join(ingredients))

    def get_names(self):
        names = []
        for line in csv.reader(open(self.corpus_path, 'r', encoding='utf-8'), delimiter=','):
            if not line[0]:
                continue  # skip header row
            names.append(line[1])
        return names

    def get_ingredients(self):
        for line in csv.reader(open(self.corpus_path, 'r', encoding='utf-8'), delimiter=','):
            if not line[0]:
                continue  # skip header row
            ingredients = line[5].split('::')
            yield [self.pre_process_ingredient(item) for item in ingredients]

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
        return re.sub(r'\d+[a-zA-Z/¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]+|\d+', '', item).strip()

    def get_or_create_model(self, path=None):
        if not path:
            path = os.path.join('..', 'created_models', 'w2v_ingredients.model')
        if not os.path.exists(path):
            self._model = Word2Vec(RecipeCorpus())
        else:
            self._names = self.get_names()
            self._model = Word2Vec.load(path)

    def save_template(self, name='w2v_ingredients.model'):
        self._model.save(os.path.join('..', 'created_models', name))

    def save_trained(self, name='trained_w2v_ingredients.kv'):
        if not self.vectors:
            raise SetupError('Model not yet trained')
        self.vectors.save(os.path.join('..', 'created_models', name))

    def load_or_train(self, name='trained_w2v_ingredients.kv'):
        path = os.path.join('..', 'created_models', name)
        if not os.path.exists(path):
            self.train()
        else:
            self.vectors = KeyedVectors.load(path, mmap='r')
            self.rec_vecs = self.calc_rec_vecs()

    def train(self, epochs=5, **kwargs):
        self._model.train(RecipeCorpus(), epochs=epochs, total_examples=self._model.corpus_count, **kwargs)
        self.vectors = self._model.wv
        self.rec_vecs = self.calc_rec_vecs()

    def calc_rec_vecs(self):
        all_vecs = []
        for y in RecipeCorpus():
            vecs = []
            for x in y:
                try:
                    vec = self.vectors[x]
                except KeyError:
                    vec = np.zeros(100)
                vecs.append(vec)
            vec_sum = sum(vecs)
            if not isinstance(vec_sum, np.ndarray) or len(vec_sum) != 100:
                vec_sum = np.zeros(100)
            all_vecs.append(vec_sum)
        return np.asarray(all_vecs)

    def cluster(self, n=3):
        self.k_means = KMeans(
            n_clusters=n, random_state=0
        ).fit(self.rec_vecs)

    def print_clusters(self):
        labels = list(self.k_means.predict(self.rec_vecs))
        labels_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            labels_dict[label].append(self._names[idx])
        print(labels_dict)

    def get_similar_to(self, recipe_name, k=5):
        name_idx = self._names.index(recipe_name)
        target_vec = self.rec_vecs[name_idx]
        reduced_rec_vecs = np.delete(self.rec_vecs, [name_idx], axis=0)
        kd_tree = KDTree(reduced_rec_vecs)
        _, indexes = kd_tree.query(target_vec, k=k)
        similar = [self._names[idx] for idx in indexes]
        print(f'Top {k} most similar recipes to {recipe_name}: \n{similar}')
        return similar

    def replace_ingredient(self, missing_ingredient, k=5):
        replacements = self.model.wv.most_similar(positive=[missing_ingredient], topn=k)
        print(f'Replacing {missing_ingredient} with: \n{replacements}')
        return replacements


class SetupError(BaseException):
    pass


if __name__ == "__main__":
    corpus = RecipeCorpus()
    corpus.get_or_create_model()
    corpus.save_template()
    corpus.load_or_train()
    corpus.save_trained()
