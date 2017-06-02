from ensemble.XiaoxuanWang import XiaoxuanWang
from utils.dataset import DataSet

d = DataSet()
slave = XiaoxuanWang(d,d.stances)

all_folds = d.stances


test_dataset = DataSet("test")
slave.dataset.articles.update(test_dataset.articles)


slave.preload_features(d.stances)
slave.prepare_final(d,test_dataset, "test.")

slave.train(d.stances)
prd = slave.predict(test_dataset.stances)
