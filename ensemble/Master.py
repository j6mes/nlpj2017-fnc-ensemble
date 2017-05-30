from scipy import spatial
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from ensemble import XiaoxuanWang
from ensemble.Classifier import Classifier
from ensemble.FNCBaseLine import binary_co_occurence, binary_co_occurence_stops, count_grams, polarity_features, \
    refuting_features, word_overlap_features
from ensemble.XiaoxuanWang import get_word2vector_f
from ensemble.XiaoxuanWang import tfidf_feature
from utils.score import report_score
import os
import pickle
class Master(Classifier):
    tfidfs = dict()
    def __init__(self,dataset):
        super().__init__(dataset)
        topo_size = (300, 8)
        self.mlpc = MLPClassifier(hidden_layer_sizes=topo_size, random_state=19940807)

        if not os.path.isfile("features/tfidf.pickle"):
            Master.tfidfs.update(self.precompute_tf_idfs(dataset))
            pickle.dump(Master.tfidfs, open("features/tfidf.pickle", "wb+"))
        else:
            tfs = pickle.load(open("features/tfidf.pickle", "rb"))
            Master.tfidfs.update(tfs)


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



    def preload_features(self, stances, ffns=list([
                                               tfidf_feature,
                                               get_word2vector_f,
                                               polarity_features,
                                               refuting_features])):

        self.fdict = self.load_feats("features/xxw.pickle",stances,ffns)


    def predict(self,data):
        lists = list(zip(*data))
        stances = lists[0]
        other_classifiers = list(zip(*lists[1:]))

        Xs, ys = self.xys(stances)

        for i, x in enumerate(Xs):
            Xs[i] = x + list(other_classifiers[i])

        pred = self.mlpc.predict(Xs)

        print(report_score(ys,pred))

        return pred

    def fit(self,data):
        lists = list(zip(*data))
        stances = lists[0]
        other_classifiers = list(zip(*lists[1:]))

        Xs,ys = self.xys(stances)

        for i,x in enumerate(Xs):
            Xs[i] = x + list(other_classifiers[i])

        self.mlpc.fit(Xs,ys)







