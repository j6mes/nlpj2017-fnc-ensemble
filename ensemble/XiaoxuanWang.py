from sklearn import feature_extraction

from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from ensemble.Classifier import Classifier
from utils.score import report_score



from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

import os
import gensim
import re
import numpy as np
import pickle
def clean(data_str):
    word_re = r'(' \
              r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
              r'(?:(?:(?:[A-Za-z0-9]+[-]+)+(?:[A-Za-z0-9]+)*))|' \
              r'(?:[$£]+[A-Za-z0-9]+)|' \
              r'(?:[A-Za-z0-9]+)|' \
              r'(?:[\!\?]+)' \
              r')'
    return re.findall(word_re, data_str)


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]






def word2vecModel():
    # more_sents = get_sents(body_set, headline_set)
    srcPath = r'GoogleNews-vectors-negative300-small.bin'
    path = os.path.abspath(srcPath)
    # load the model
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    # model = gensim.models.Word2Vec.load(path, binary=True)
    # model.train(more_sents)
    return model

model = word2vecModel()

def get_word2vector_f(id,headline, body):
    feature =[]
    # word_re = r'(' \
    #           r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
    #           r'(?:(?:(?:[A-Za-z0-9]+[-]+)+(?:[A-Za-z0-9]+)*))|' \
    #           r'(?:[$£]+[A-Za-z0-9]+)|' \
    #           r'(?:[A-Za-z0-9]+)|' \
    #           r'(?:[\!\?]+)' \
    #           r')'

    clean_headline = clean(headline)
    clean_body = clean(body)

    dim = 300

    headline_word_count = 0
    word_vec_head = np.zeros(dim)
    for i, word_head in enumerate(clean_headline):
        try:
            word_vec_head += model[word_head]
            headline_word_count += 1
        except KeyError:
            # ignore if word not in vocabulary
            continue

    body_word_count = 0
    word_vec_body = np.zeros(dim)
    for j, word_body in enumerate(clean_body):
        try:
            word_vec_body += model[word_body]
            body_word_count += 1
        except KeyError:
            # ignore if word not in vocabulary
            continue


    word_vec_head = word_vec_head / headline_word_count
    word_vec_body = word_vec_body / body_word_count


    feature.extend(word_vec_head)
    feature.extend(word_vec_body)

    # print('feature size: {}'.format(len(feature)))
    return feature

def word_overlap_features(headline_str, body_str):


    clean_headline = clean(headline_str)
    clean_body = clean(body_str)
    feature = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

    return feature

def refuting_features(id,headline_str, body_str):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    clean_headline = clean(headline_str)
    feature = [1 if word in clean_headline else 0 for word in _refuting_words]

    return feature

def polarity_features(id,headline_str, body_str):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):

        return sum([t in _refuting_words for t in text]) % 2


    clean_headline = clean(headline_str)
    clean_body = clean(body_str)
    feature = []
    feature.append(calculate_polarity(clean_headline))
    feature.append(calculate_polarity(clean_body))

    return feature


# hand_features
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features

def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def hand_features(headline_str, body_str):
    feature = []
    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features


    feature.append(binary_co_occurence(headline_str, body_str)
             + binary_co_occurence_stops(headline_str, body_str)
             + count_grams(headline_str, body_str))

    return feature


# def word2vecFeature(headline_str, body_str):
#     X = []
#     model = word2vecModel()
#     for (headline, body) in zip(headline_set, body_set):
#         feature = []
#         # clean_headline = clean(headline)
#         # clean_body = clean(body)
#         clean_headline = get_tokenized_lemmas(clean_headline)
#         clean_body = get_tokenized_lemmas(clean_body)
#
#         dim = len(model['cat'])
#
#         headline_word_count = 0
#         word_vec_head = np.zeros(dim)
#         for i, word_head in enumerate(clean_headline):
#             try:
#                 word_vec_head += model[word_head]
#                 headline_word_count += 1
#             except KeyError:
#                 # ignore if word not in vocabulary
#                 continue
#
#         body_word_count = 0
#         word_vec_body = np.zeros(dim)
#         for j, word_body in enumerate(clean_body):
#             try:
#                 word_vec_body += model[word_body]
#                 body_word_count += 1
#             except KeyError:
#                 # ignore if word not in vocabulary
#                 continue
#
#
#         word_vec_head = word_vec_head / headline_word_count
#         word_vec_body = word_vec_body / body_word_count
#
#
#         feature.extend(word_vec_head)
#         feature.extend(word_vec_body)
#
#         X.append(feature)
#
#     return X




class XiaoxuanWang(Classifier):
    def __init__(self,dataset,train):
        super().__init__(dataset,train)
        topo_size = (300, 8)
        self.mlpc = MLPClassifier(hidden_layer_sizes=topo_size, random_state=19940807)

        self.tfidfs = dict()


        if not os.path.isfile("features/tfidf.pickle"):
            self.tfidfs.update(self.precompute_tf_idfs(dataset))
            pickle.dump(self.tfidfs, open("features/tfidf.pickle", "wb+"))
        else:
            tfs = pickle.load(open("features/tfidf.pickle", "rb"))
            self.tfidfs.update(tfs)


    def precompute_tf_idfs(self,dataset):
        headlines = []
        bodies = []
        for stance in dataset.stances:
            headlines.append(stance['Headline'])
            bodies.append(dataset.articles[stance['Body ID']])

        train_set = []
        train_set.extend(headlines)
        train_set.extend(bodies)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_train = tfidf_vectorizer.fit_transform(train_set).todense()

        sim_scores = dict()
        for x, stance in tqdm(enumerate(dataset.stances)):
            sim_scores[stance['Stance ID']] = 1 - spatial.distance.cosine(tfidf_train[x], tfidf_train[x + len(bodies)])

        return sim_scores


    def predict(self,data):
        Xs,ys = self.xys(data)
        prd = self.mlpc.predict(Xs)
        return prd


    def train(self,data):
        Xs,ys = self.xys(data)
        self.mlpc.fit(Xs, ys)

    def load_w2v(self):
        pickle.load(open("features/tfidf.pickle","rb"))

    def delete_big_files(self):
        del self.tfidfs

    def tfidf_feature(self,id, headline, body):
        return [self.tfidfs[id]]

    def preload_features(self, stances, ffns=list([
                                               get_word2vector_f,
                                               polarity_features,
                                               refuting_features])):
        ffns.append(self.tfidf_feature)
        self.fdict = self.load_feats("features/xxw.pickle",stances,ffns)




