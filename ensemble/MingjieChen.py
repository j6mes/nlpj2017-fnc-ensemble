from ensemble.Classifier import Classifier
from original.Mingjie_Chen import MLP


class MingjieChen(Classifier):
    def __init__(self,data,train):
        super().__init__(data,train)
        self.mlp = MLP()