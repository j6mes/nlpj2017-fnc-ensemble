import sys
import io
import nltk
import os

current_path = os.path.dirname(os.path.abspath(__file__))
folder1_path = os.path.join(current_path, 'fakenewschallenge-teaching')
folder2_path = os.path.join(current_path, 'fakenewschallenge-teaching', 'fnc-1')
folder3_path = os.path.join(current_path, 'fakenewschallenge-teaching', 'splits')
folder4_path = os.path.join(current_path, 'fakenewschallenge-teaching', 'utils')
folder5_path = os.path.join(current_path, 'pjslib')
sys.path.append(folder1_path)
sys.path.append(folder2_path)
sys.path.append(folder3_path)
sys.path.append(folder4_path)
sys.path.append(folder5_path)
# from score import *
from utils.score import report_score
# print ("sys.path: {}".format(sys.path))

import re
import pickle
import numpy as np
from sklearn import svm
import collections
from sklearn.neural_network import MLPClassifier

import gensim
import numpy as np
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
folder1_path = os.path.join(current_path, 'pjslib')
sys.path.append(folder1_path)
from nltk.corpus import wordnet

class Feature:
    def __init__(self, word2vector):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vector, binary=True)
        self.not_found_word_set = set()

    def word2vector_f(self, word_list):
        word_vector_array = None
        for i, word in enumerate(word_list):
            if word_vector_array is None:
                try:
                    word_vector_array = np.copy(self.model[word])
                except KeyError:
                    pass
            else:
                try:
                    word_vector_array += self.model[word]
                except KeyError:
                    pass
        return word_vector_array


    def word2vector_f_average(self, word_list):
        word_vector_array = None
        word_count = 0
        for i, word in enumerate(word_list):
            if word_vector_array is None:
                try:
                    word_vector_array = np.copy(self.model[word])
                    word_count += 1
                except KeyError:
                    self.not_found_word_set.add(word)

            else:
                try:
                    word_vector_array += self.model[word]
                    word_count += 1
                except KeyError:
                    self.not_found_word_set.add(word)


        if word_vector_array is not None:
            word_vector_array = word_vector_array / word_count
        return word_vector_array

    def logging_not_found_word(self):
        return
        #for word in self.not_found_word_set:
        #    logger1.info("{} not found in word_vector".format(word))

    def compare_synonyms_feature(self, body_word_list, headline_word_list):

        def get_synonyms_antonyms_set(word):

            def get_synonyms_temp_antonyms(word):
                synonyms = []
                temp_antonyms = []
                for syn in wordnet.synsets(word):
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            temp_antonyms.append(l.antonyms()[0].name())
                return synonyms, temp_antonyms

            def get_full_antonyms(temp_antonyms):
                antonyms = []
                for word in temp_antonyms:
                    for syn in wordnet.synsets(word):
                        for l in syn.lemmas():
                            antonyms.append(l.name())
                return antonyms

            # :::get_synonyms_antonyms_set
            synonyms, temp_antonyms = get_synonyms_temp_antonyms(word)
            antonyms = get_full_antonyms(temp_antonyms)
            return synonyms, antonyms

        # :::compare_synonyms_feature
        body_synonyms_set = set()
        body_antonyms_set = set()
        headline_synonyms_set = set()
        headline_antonyms_set = set()
        #
        for word in body_word_list:
            synsets, antsets = get_synonyms_antonyms_set(word)
            body_synonyms_set.update(set(synsets))
            body_antonyms_set.update(set(antsets))
        #

        #
        for word in headline_word_list:
            synsets, antsets = get_synonyms_antonyms_set(word)
            headline_synonyms_set.update(set(synsets))
            headline_antonyms_set.update(set(antsets))
        #
        overlap_s_s_set = body_synonyms_set.intersection(headline_synonyms_set)
        overlap_s_a_set = body_synonyms_set.intersection(headline_antonyms_set)
        overlap_a_s_set = body_antonyms_set.intersection(headline_synonyms_set)
        overlap_a_a_set = body_antonyms_set.intersection(headline_antonyms_set)


        feature_count_array = np.array([len(overlap_s_s_set), len(overlap_s_a_set),
                                        len(overlap_a_s_set), len(overlap_a_a_set)])
        print("feature_count_array: {}".format(feature_count_array))
        # normalize
        word_list_len = len(body_word_list) + len(headline_word_list)
        word_list_len_log = np.log2(word_list_len)
        feature_count_array = feature_count_array/word_list_len_log
        # #
        # print ("body_synonyms_set: {}".format(body_synonyms_set))
        # print("body_antonyms_set: {}".format(body_antonyms_set))
        # print("headline_synonyms_set: {}".format(headline_synonyms_set))
        # print("headline_antonyms_set: {}".format(headline_antonyms_set))
        # print("feature_count_array: {}".format(feature_count_array))
        # sys.exit(0)

        return feature_count_array



    def punctuation_mark_feature(self, word_list):

        _punctuation_mark = [
            '?',
            '!',
        ]
        feature_count_array = np.zeros(len(_punctuation_mark))
        for i, check_word in enumerate(_punctuation_mark):
            words_count = word_list.count(check_word)
            feature_count_array[i] = words_count

        # # normalize
        # word_list_len = len(word_list)
        # feature_count_array /= word_list_len
        # #
        # logger2.info("================================================")
        # logger2.info("punctuation_mark_feature")
        # logger2.info("word_list: {}".format(word_list))
        # logger2.info("feature_count_array: {}".format(feature_count_array))
        # logger2.info("================================================")

        return feature_count_array


    def disagree_feature(self, word_list):
        _disagree_words = [
            'no',
            'hasn\'t',
            'haven\'t',
            'not',
            'didn\'t',
            'aren\'t',
            'never',
            'disagree',
            'disagrees',
            'disagreed',
            'agree',
            'agrees',
            'agreed',
            'isn\'t',
            'wasn\'t',
            'weren\'t',
            'wouldn\'t',
            'shouldn\'t',
            'can\'t',
            'couldn\'t',
            'cannot',
            'won\'t',
            'fake'
        ]
        feature_count_array = np.zeros(len(_disagree_words))
        for i, check_word in enumerate(_disagree_words):
            words_count = word_list.count(check_word)
            feature_count_array[i] = words_count

        # normalize
        word_list_len = len(word_list)
        feature_count_array /= word_list_len
        #



        return feature_count_array

    def refuting_feature(self, word_list):
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
        refuting_words_count = 0
        for word in word_list:
            if word in _refuting_words:
                refuting_words_count += 1


        return np.array([refuting_words_count])



    def n_gram_overlap_feature(self, body_word_list, headline_word_list):

        def create_n_gram_tuple_list(n, word_list):
            tuple_list = []
            max_i = len(word_list) - n
            for i, word in enumerate(word_list):
                if i> max_i:
                    continue
                else:
                    tuple1 = tuple(word_list[i:i+n])
                tuple_list.append(tuple1)
            return tuple_list

        def count_overlap(word_list1, word_list2):
            overlap = 0
            for word_tuple in word_list1:
                overlap += word_list2.count(word_tuple)
            return overlap

        #:::n_gram_overlap_feature
        N_GRAM_LIST = [1,2,3,4,5,6,7]
        overlap_count_array = np.zeros(len(N_GRAM_LIST))
        for i, n in enumerate(N_GRAM_LIST):
            body_ngram_word_list = create_n_gram_tuple_list(n, body_word_list)
            headline_ngram_word_list = create_n_gram_tuple_list(n, headline_word_list)
            n_overlap = count_overlap(body_ngram_word_list, headline_ngram_word_list)
            overlap_count_array[i] = n_overlap

        # normalize
        word_list_len = len(body_word_list) + len(headline_word_list)
        overlap_count_array  /= word_list_len
        #



        return overlap_count_array

class FncClassifier:
    def __init__(self, is_PCA=False, n_components=300, is_nltk_token = False ):
        self.stopword_set = set()
        self.create_stop_words()
        # SVM
        self.svm_class_list = []
        self.svm_feature_list = []
        self.svm_clf = svm.SVC(kernel='rbf')
        # self.svm_clf = svm.SVC(kernel = 'linear')
        # self.svm_clf = svm.SVC(kernel = 'poly')
        #
        # MLP
        # hidden_layer_sizes=(100, 4)
        # hidden_layer_sizes=(200, 2) = SHIT
        # hidden_layer_sizes=(50, 2) = 55.70
        # self.mlp_clf = MLPClassifier(hidden_layer_sizes = (100, 2), random_state=1) ---- very good
        # self.mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (100, 2), random_state=1)
        # self.mlp_clf = MLPClassifier(hidden_layer_sizes = (100, 2), random_state=1)
        #


        # SWTICH
        self.FILTER_STOP_WORDS = True
        #
        self.mlp_hidden_layer_sizes_list = []
        self.mlp_result = []

        # PCA
        self.is_PCA = is_PCA
        if self.is_PCA:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=n_components)

        #nltk
        self.is_nltk_token = is_nltk_token
        if self.is_nltk_token:
            import nltk

    def examine_features(self):

        # manually check the training data
        data_set = self.training_data
        examine_feature_dict = collections.defaultdict(lambda: [])
        for i, details in enumerate(data_set):
            stance = details['Stance']
            if stance == 'unrelated':
                continue
            head_line = details['Headline']
            body_id = details['Body ID']
            article_body = self.dataset.articles[body_id]
            examine_feature_dict[stance].append((head_line, article_body))

        # write the examine_feature_dict to file
        folder_name = 'features_examine'
        folder_path = os.path.join(current_path, folder_name)
        for stance, stance_list in examine_feature_dict.items():
            file_name = stance + '.txt'
            file_path = os.path.join(folder_path, file_name)

            # head_line and body
            with open(file_path, 'w', encoding='utf-8') as f:
                for head_line, body in stance_list:
                    f.write("head_line: \n" + head_line + '\n')
                    f.write("--------------------------------------\n")
                    f.write("body: \n" + body + '\n')
                    f.write("--------------------------------------\n\n\n")

            # head line
            head_line_file_name = stance + '_headline.txt'
            head_line_file_path = os.path.join(folder_path, head_line_file_name)
            with open(head_line_file_path, 'w', encoding='utf-8') as f:
                for head_line, body in stance_list:
                    f.write("\n" + head_line + '\n')


    def set_mlp(self, hidden_layer_sizes):
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=1)

    def load_feature_generator(self, feature_generator):
        self.feature_generator = feature_generator

    def create_stop_words(self):
        with open('stoplist.txt', 'r') as f:
            for line in f:
                self.stopword_set.add(line.strip())

    def filter_stop_words(self, word_list):
        new_word_list = []
        for word in word_list:
            if word in self.stopword_set:
                continue
            else:
                new_word_list.append(word)
        return new_word_list

    def get_feature_for_stance(self, id,head_line,article_body):


        # CLEARN THE DATA ======================================================
        head_line = head_line.replace(r'’', r"'")
        article_body = article_body.replace(r'’', r"'")
        # ======================================================================

        # body
        if not self.is_nltk_token:
            word_re = r'(' \
                      r'(?:[A-Za-z0-9]+[\']+[A-Za-z0-9]+)|' \
                      r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\.]+)|' \
                      r'(?:[A-Za-z0-9]+[\-]+[A-Za-z0-9]+[\'])|' \
                      r'(?:[A-Za-z0-9]+[\-]+[A-Za-z0-9]+[\'])|' \
                      r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
                      r'(?:(?:(?:[A-Za-z0-9]+[-]+)+(?:[A-Za-z0-9]+)*))|' \
                      r'(?:[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
                      r'(?:[$£]+[A-Za-z0-9]+)|' \
                      r'(?:[A-Za-z0-9]+)|' \
                      r'(?:[\!\?]+)' \
                      r')'
        # word_re = r'[A-Za-z0-9]+[\']+[A-Za-z0-9]+|[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\.]+|[A-Za-z0-9]+[\']+|[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+|[A-Za-z0-9]+[\-]+[A-Za-z0-9]+|[A-Za-z0-9]+'

        # body
        if not self.is_nltk_token:
            body_word_list = re.findall(word_re, article_body)
        else:
            body_word_list = nltk.word_tokenize(article_body)
        body_word_list = [word.lower() for word in body_word_list]
        if self.FILTER_STOP_WORDS:
            body_word_list = self.filter_stop_words(body_word_list)
        #

        # headline
        if not self.is_nltk_token:
            headline_word_list = re.findall(word_re, head_line)
        else:
            headline_word_list = nltk.word_tokenize(head_line)
        headline_word_list = [word.lower() for word in headline_word_list]
        if self.FILTER_STOP_WORDS:
            headline_word_list = self.filter_stop_words(headline_word_list)
        #

        # print ("word_list : {}".format(word_list))

        # print ("body_word_list: {}".format(body_word_list))
        # print("headline_word_list: {}".format(headline_word_list))
        # sys.exit(0)

        all_feature_vector_list = []

        # (0.) word2vector
        body_feature_vector = self.feature_generator.word2vector_f_average(body_word_list)
        headline_feature_vector = self.feature_generator.word2vector_f_average(headline_word_list)
        if headline_feature_vector is None or body_feature_vector is None:  # skip if headline has all unknown words
            feature_vector = None
            return feature_vector
        all_feature_vector_list.append(body_feature_vector)
        all_feature_vector_list.append(headline_feature_vector)
        #

        # (1.) refuting_feature
        body_refuting = self.feature_generator.refuting_feature(body_word_list)
        headline_refuting = self.feature_generator.refuting_feature(headline_word_list)
        all_feature_vector_list.append(body_refuting)
        all_feature_vector_list.append(headline_refuting)
        #

        # punctuation_mark_feature
        body_punctuation_mark = self.feature_generator.punctuation_mark_feature(body_word_list)
        headline_punctuation_mark = self.feature_generator.punctuation_mark_feature(headline_word_list)
        all_feature_vector_list.append(body_punctuation_mark)
        all_feature_vector_list.append(headline_punctuation_mark)
        #

        # # disagree_feature
        # body_disagree = self.feature_generator.disagree_feature(body_word_list)
        # headline_disagree = self.feature_generator.disagree_feature(headline_word_list)
        # all_feature_vector_list.append(body_disagree)
        # all_feature_vector_list.append(headline_disagree)
        # #

        # n_gram_overlap_feature
        n_gram_overlap_feature = self.feature_generator.n_gram_overlap_feature(body_word_list, headline_word_list)
        all_feature_vector_list.append(n_gram_overlap_feature)
        #


        # # compare_synonyms_feature
        # synonyms_feature = self.feature_generator.compare_synonyms_feature(body_word_list, headline_word_list)
        # all_feature_vector_list.append(synonyms_feature)
        # #

        feature_vector = np.array([])
        for feature_vector_now in all_feature_vector_list:
            feature_vector = np.concatenate((feature_vector, feature_vector_now))


        return feature_vector

    def np_save_feature_vector(self, file_path, feature_vector):
        np.save(file_path, feature_vector)

    def save_features_to_local_file(self, target='train'):
        if target == 'train':
            folder = 'train_feature_vectors'
            data_set = self.training_data
        elif target == 'dev':
            folder = 'dev_feature_vectors'
            data_set = self.dev_data
        elif target == 'test':
            folder = 'test_feature_vectors'
            data_set = self.test_data
        else:
            print("feature save target folder error!")
            sys.exit(0)

        print("Save {} feature vectors to local start!".format(target))
        file_path_list = []
        PCA_vector_list = []
        for i, details in enumerate(data_set):
            stance, feature_vector = self.get_feature_for_stance(details)
            #
            if i == 0:
                print("feature_vector_len_before_PCA: {}".format(len(feature_vector)))
            #

            if feature_vector is None:
                print(" {}th data feature vector none[{}]".format(i, target))
            file_name = "id#{}#stance#{}#".format(i, stance)
            file_path = os.path.join(folder, file_name)
            # print ("stance: ", stance)
            # print("feature_vector: ", feature_vector)
            # sys.exit(0)
            if not self.is_PCA:
                self.np_save_feature_vector(file_path, feature_vector)
            else:
                file_path_list.append(file_path)
                PCA_vector_list.append(feature_vector)
                PCA_vector_array = np.array(PCA_vector_list)
        # apply pca and save file
        if self.is_PCA:
            if target == 'train':
                self.pca.fit(PCA_vector_array)
            pca_feature_vector_array = self.pca.transform(PCA_vector_array)
            for file_path, feature_vector in zip(file_path_list, pca_feature_vector_array):
                self.np_save_feature_vector(file_path, feature_vector)
        #

        print("Save all feature vectors to local successful!")



    def mlp_train(self):
        # load train vectors
        folder = 'train_feature_vectors'
        file_path_list = os.listdir(folder)
        file_path_list = file_path_list
        class_list = []
        feature_list = []
        total = len(file_path_list)
        for i, file_path in enumerate(file_path_list):
            file_name = os.path.basename(file_path)
            file_path = os.path.join(folder, file_path)
            stance = re.findall(r'stance#(.*)#', file_name)
            if stance:
                stance = stance[0]
            else:
                continue
            feature_vector = np.load(file_path)
            if (feature_vector == np.array(None)).all():
                continue
            #
            class_list.append(stance)
            feature_list.append(feature_vector)
            # print ("push {}/{} data vector to svm list".format(i, total))
            #
        # train svm
        print("Start training MLP!")
        self.mlp_clf.fit(feature_list, class_list)
        print("Training MLP finish!")
        # save svm to local
        pickle.dump(self.mlp_clf, open("mlp_classifier", "wb"))


    def mlp_predict(self,data):
        # load the svm classifier
        mlp_clf = pickle.load(open("mlp_classifier", "rb"))
        #
        total_dev_data = len(data)
        folder = 'test_feature_vectors'
        file_path_list = os.listdir(folder)
        correct = 0
        print("Testing Set!")
        actual_list = []
        predicted_list = []
        for i, file_path in enumerate(file_path_list):
            file_name = os.path.basename(file_path)
            file_path = os.path.join(folder, file_path)
            stance = re.findall(r'stance#(.*)#', file_name)[0]
            feature_vector = np.load(file_path)
            feature_vector = feature_vector.reshape(1, -1)
            if (feature_vector == np.array(None)).all():
                print("feature_vector is None!")
                sys.exit(0)
            # print("file_path: {}".format(file_path))
            # print ("feature_vector: {}".format(feature_vector))
            clf_stance = mlp_clf.predict(feature_vector)
            if stance == clf_stance:
                correct += 1
            actual_list.append(stance)
            predicted_list.append(clf_stance)
            # print ("add correct, dev data id :{}".format(i))

        # save accuracy
        Accuracy = report_score(actual_list, predicted_list)
        self.mlp_result.append(Accuracy)
        #

    def mlp_print_result(self, mode = 'dev'):
        file_name = "{}_result.txt".format(mode)
        print("self.mlp_result: {}".format(self.mlp_result))
        self.mlp_result = [float(x) for x in self.mlp_result]
        print_list = list(zip(self.mlp_hidden_layer_sizes_list, self.mlp_result))
        print_list = sorted(print_list, key=lambda x: x[1], reverse=True)
        with open(file_name, 'w') as f:
            for tuple1 in print_list:
                size = str(tuple1[0])
                accuracy = str(tuple1[1])
                f.write(size + ',' + accuracy + '\n')