import sys
import numpy as np
from scipy import spatial

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ensemble.FNCBaseLine import FNCBaseLine
from ensemble.Master import Master
from ensemble.XiaoxuanWang import XiaoxuanWang
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from upperbound import compute_ub
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
import pickle
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

    master_classifier = None

    train = dict()
    test = dict()

    ids = list(range(len(folds)))
    all_folds = np.hstack(tuple([fold_stances[i] for i in ids]))

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        train[fold] = np.hstack(tuple([fold_stances[i] for i in ids]))
        test[fold] = fold_stances[fold]

    slave_classifiers = [FNCBaseLine,XiaoxuanWang]


    slv_predicted = dict()
    master_train = dict()

    import os

    if not os.path.isfile("features/master_train.pickle"):
        for fold in tqdm(fold_stances):
            slv_predicted[fold] = []
            master_train[fold] = []
            for slv in tqdm(slave_classifiers):
                cls = slv(d)
                cls.preload_features(d.stances)
                cls.train(train[fold])

                slv_predicted[fold].append([LABELS.index(p) for p in cls.predict(test[fold])])
                del cls
            master_train[fold].extend(zip(test[fold], *slv_predicted[fold]))

        pickle.dump(master_train, open("features/master_train.pickle","wb+"))
    else:
        master_train = pickle.load(open("features/master_train.pickle","rb"))

    slaves = []
    if not os.path.isfile("features/slaves.pickle"):
        for slv in tqdm(slave_classifiers):
            cls = slv(d)
            cls.preload_features(d.stances)
            cls.train(all_folds)
            slaves.append(cls)
        pickle.dump(slaves, open("features/slaves.pickle","wb+"))
    else:
        slaves = pickle.load(open("features/slaves.pickle","rb"))

    print("UPPER BOUND:::")
    compute_ub(slaves,hold_out_stances)

    mdata = []
    for fold in fold_stances:
        mdata.extend(master_train[fold])
    master = Master(d)
    master.preload_features(d.stances)
    master.fit(mdata)

    slv_predicted_holdout = []
    for slave in slaves:
        slv_predicted_holdout.append([LABELS.index(p) for p in slave.predict(hold_out_stances)])

    final_predictions = master.predict(zip(hold_out_stances,*slv_predicted_holdout))
    report_score(master.xys(hold_out_stances)[1],final_predictions)