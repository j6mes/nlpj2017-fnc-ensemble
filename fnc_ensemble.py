import sys
import numpy as np
from scipy import spatial

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ensemble.GiorgosMyrianthous import GiorgosMyrianthous
from ensemble.JiashuPu import JiashuPu
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

    slave_classifiers = [FNCBaseLine,XiaoxuanWang,JiashuPu,GiorgosMyrianthous]

    slv_predicted = dict()
    master_train = dict()

    import os

    if not os.path.isfile("features/master_train.pickle"):
        for fold in tqdm(fold_stances):
            slv_predicted[fold] = []
            master_train[fold] = []
            for slv in tqdm(slave_classifiers):
                cls = slv(d,all_folds)
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
            print("Training classifier" + str(type(slv)))
            cls = slv(d)
            cls.preload_features(d.stances)
            cls.train(all_folds)
            slaves.append(cls)
            cls.delete_big_files()
        pickle.dump(slaves, open("features/slaves.pickle","wb+"))
    else:
        slaves = pickle.load(open("features/slaves.pickle","rb"))


    for slave in slaves:
        print("Loading features for slave " + str(type(slave)))
        slave.preload_features(d.stances)
        slave.load_w2v()


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