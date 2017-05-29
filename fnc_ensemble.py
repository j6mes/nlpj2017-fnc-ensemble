import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from ensemble.FNCBaseLine import FNCBaseLine
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
def generate_features(stances,dataset,name):
    return [[1,1],[0,0]],[0,1]


class Slv1Cls:
    def __init__(self):
        self.a = GradientBoostingClassifier()

    def features(self,datum):
        h_words = datum['Headline'].lower().split()
        b_words = d.articles[datum['Body ID']].lower().split()

        wc = 0
        for word in h_words:
            wc += word in b_words

        return [wc],LABELS.index(datum['Stance'])

    def xys(self,data):
        Xs = []
        ys = []
        for datum in data:
            fts = self.features(datum)
            Xs.append(fts[0])
            ys.append(fts[1])

        return Xs,ys

    def train(self,data):
        Xs,ys = self.xys(data)
        self.a.fit(Xs,ys)

    def predict(self,data):
        Xs,ys = self.xys(data)
        prd = self.a.predict(Xs)
        print(score_submission([LABELS[i] for i in ys],[LABELS[i] for i in prd]))
        return prd

class Slv2Cls:
    def __init__(self):
        self.a = LinearRegression()

    def features(self,datum):
        h_words = datum['Headline'].lower().split()
        b_words = d.articles[datum['Body ID']].lower().split()

        wc = 0
        for word in h_words:
            wc += word in b_words

        return [wc],LABELS.index(datum['Stance'])

    def xys(self,data):
        Xs = []
        ys = []
        for datum in data:
            fts = self.features(datum)
            Xs.append(fts[0])
            ys.append(fts[1])

        return Xs,ys

    def train(self,data):
        Xs,ys = self.xys(data)
        self.a.fit(Xs,ys)

    def predict(self,data):
        Xs,ys = self.xys(data)
        prd = self.a.predict(Xs)
        print(score_submission([LABELS[i] for i in ys],[LABELS[int(i)] for i in prd]))
        return prd


if __name__ == "__main__":
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=2)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    Xs = dict()
    ys = dict()

    fb = FNCBaseLine(d)
    fb.preload_features(d.stances)

    slave_classifiers = [fb]
    master_classifier = None

    train = dict()
    test = dict()

    master_train = []

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        train[fold] = np.hstack(tuple([fold_stances[i] for i in ids]))
        test[fold] = fold_stances[fold]

    fold = 0

    slv_predicted = []
    for slave in slave_classifiers:
        s = slave
        s.train(train[fold])

        slv_predicted.append(s.predict(test[fold]))
        print(slv_predicted)
    master_train.extend(zip(test[fold],*slv_predicted))

