import itertools
import math
import re
from collections import Counter

import pandas as pd
import os


class TfIdf(object):

    def __init__(self):
        self.corpus = []

    @staticmethod
    def from_list(list_obj):
        tf_idf = TfIdf()
        tf_idf.corpus = list_obj
        return tf_idf

    @staticmethod
    def from_dict(dict_obj):
        tf_idf = TfIdf()
        tf_idf.corpus = itertools.chain(*[v for v in dict_obj.values()])
        return tf_idf

    @staticmethod
    def pre_process_corpus(corpus):
        processed_documents = []
        r = re.compile('[^a-zA-Z ]')
        for document in corpus:
            document = r.sub('', document.lower())
            processed_documents.append(document)
        return processed_documents

    @staticmethod
    def calculate_document_freqs(document):
        doc_word_mask = {}
        terms = document.split()
        local_rankings = Counter(terms)
        tf_tuples = []
        for term in terms:
            tf_tuples.append((term, local_rankings[term] / len(terms)))
            doc_word_mask[term] = True
        return doc_word_mask, tf_tuples

    @staticmethod
    def get_tf_idf_vector(term_freq_vector, num_docs, doc_freqs):
        tf_idf_vector = []
        for term, term_freq in term_freq_vector:
            tf_idf_vector.append(term_freq * (math.log(num_docs / doc_freqs[term])))
        return tf_idf_vector

    def perform_tf_idf(self, corpus):
        corpus = self.pre_process_corpus(corpus)
        term_freqs = []
        doc_freqs = {}
        tf_idfs = []
        for document in corpus:
            doc_word_mask, tf_tuples = self.calculate_document_freqs(document)
            doc_freqs.update({term: doc_freqs.get(term, 0) + 1 for term in doc_word_mask.keys()})
            term_freqs.append(tf_tuples)
        num_docs = len(corpus)
        for term_freq_vector in term_freqs:
            tf_idfs.append(self.get_tf_idf_vector(term_freq_vector, num_docs, doc_freqs))
        return tf_idfs


if __name__ == "__main__":
    _df = pd.read_csv(os.path.join('data', 'recipe_details.csv'), index_col=0)
    names = _df['name'].tolist()
    tf_idf_obj = TfIdf()
    _tf_idf = tf_idf_obj.perform_tf_idf(names)
    print(213)
