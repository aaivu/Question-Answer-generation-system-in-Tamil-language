import pickle
import warnings

import scipy
import sklearn_crfsuite
import snowballstemmer
import csv
import re

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import flat_accuracy_score
from sklearn_crfsuite.metrics import flat_f1_score
from src.Code.Utils.tamilutils import findinflectionsuffix
from src.Code.Utils.lists import gazetteers


def lookup_search(lookup, stem_word):
    for key, value in gazetteers.gazetteer_list.items():
        for v in value:
            part_names = v.split(" ")
            if lookup in part_names:
                return key
    return 'None'


def is_year_format(word):
    return re.match('^[1-2][0-9]{3}(?:இல்|ல்|ஆம்|ம்){0,1}', word) != 'None'


def check_if_ordinal_numeric_word(word):
    return word in gazetteers.previous_word.get('ORDINAL')


def check_prefix_word(word):
    if word in gazetteers.previous_word.get('TER'):
        return 'TER'
    if word in gazetteers.previous_word.get('COU'):
        return 'COU'
    if word in gazetteers.previous_word.get('ORG'):
        return 'ORG'
    if word in gazetteers.previous_word.get('TIME'):
        return 'TIME'

    return 'None'


def check_if_word_ends_with_comma(word):
    return word.endswith(',')


def word_is_conjunction(word):
    common_conj_words = ['மற்றும்', 'ஆகிய', 'ஆகியன', 'போன்ற', 'போன்றன', 'போன்றவை', 'ஆகியவை',
                         'என்பன', 'என்பவை', 'ஆகியோர்', 'போன்றோர்']
    return word in common_conj_words


def check_if_word_is_number(word):
    if word.isdigit():
        return True
    for key, value in gazetteers.numbers.items():
        for v in value:
            if word == v:
                return True
    return False


def check_clue_word_list(word):
    stemmer = snowballstemmer.stemmer('tamil')
    stem_word = stemmer.stemWords(word.split())
    for key in gazetteers.clue_words:
        if word in gazetteers.clue_words.get(key) or stem_word in gazetteers.clue_words.get(key):
            return key

    return 'NONE'


def get_stem_word(word):
    stemmer = snowballstemmer.stemmer('tamil')
    return stemmer.stemWords(word.split())


def is_word_contains_hyphen(word):
    return re.search(r'-', word) != 'None'


def is_hyphen(word):
    return word == '-'


def gazetteer_check(word):
    stem_word = get_stem_word(word)
    return lookup_search(word, stem_word)


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        # 'word[-2:]': word[-2:],
        'suffix': findinflectionsuffix(word),
        'isnumber': check_if_word_is_number(word),
        'postag': postag,
        'clueword': check_clue_word_list(word),
        'gazetteerword': gazetteer_check(word),
        'isordinal': check_if_ordinal_numeric_word(word),
        'isyearformat': is_year_format(word),
        'stemword': get_stem_word(word)
    }

    if i > 0:
        pword1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        plabel1 = sent[i - 1][2]
        features.update({
            '-1:word': pword1,
            # '-1:word[-2:]': word[-2:],
            '-1:postag': postag1,
            '-1:clueword': check_clue_word_list(pword1),
            '-1:isnumber': check_if_word_is_number(pword1),
            '-1:gazetteerword': gazetteer_check(pword1),
            '-1:previouslabel': plabel1,
            '-1:isordinal': check_if_ordinal_numeric_word(pword1),
            '-1:isyearformat': is_year_format(pword1),
            '-1:isconjunction': word_is_conjunction(pword1),
            '-1:endwithcomma': check_if_word_ends_with_comma(pword1),
            '-1:stemword': get_stem_word(pword1),
        })
        # if i > 1:
        #     pword2 = sent[i-2][0]
        #     postag2 = sent[i-2][1]
        #     plabel2 = sent[i-2][2]
        #     features.update({
        #         '-2:word': pword2,
        #         '-2:postag': postag2,
        #         '-2:previouslabel': plabel2,
        #         # '-2:gazetteerword': gazettercheck(pword2),
        #         '-2:isyearformat': isyearformat(pword2),
        #         '-2:clueword': checkcluewordlist(pword2),
        #         '-2:endwithcomma': checkifwordendswithcomma(pword2),
        #         '-2:stemword': getstemword(pword2)
        #     })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        nword1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word': nword1,
            '+1:postag': postag1,
            '+1:clueword': check_clue_word_list(nword1),
            '+1:gazetteerword': gazetteer_check(nword1),
            '+1:isnumber': check_if_word_is_number(nword1),
            '+1:isconjunction': word_is_conjunction(nword1),
            '+1:endwithcomma': check_if_word_ends_with_comma(nword1),
            '+1:stemword': get_stem_word(nword1)
        })
        # if i < len(sent) - 2:
        #     nword2 = sent[i+2][0]
        #     postag2 = sent[i+2][1]
        #     features.update({
        #         '+2:word': nword2,
        #         '+2:postag': postag2,
        #         # '+2:gazetteerword': gazettercheck(nword2),
        #         '+2:clueword': checkcluewordlist(nword2),
        #         '+2:isconjunction': wordisaconjunction(nword2),
        #         '+2:endwithcomma': checkifwordendswithcomma(nword2),
        #         '+2:stemword': getstemword(nword2)
        #     })

    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def train_and_test():
    sentences = read_conll_as_list()
    X = [sent2features(sentence) for sentence in sentences]
    y = [sent2labels(sentence) for sentence in sentences]
    label_category = ['B-NAME', 'I-NAME', 'B-YEAR', 'B-MON', 'I-MON', 'B-COU', 'I-COU', 'B-COM', 'I-COM', 'B-CITY',
                      'I-CITY', 'B-NORP', 'B-EVE', 'I-EVE', 'B-LOC', 'I-LOC', 'B-TRO', 'I-TRO', 'B-NUM', 'B-PER',
                      'B-CON', 'B-ORG', 'B-SEC', 'I-SEC', 'B-DES', 'I-DES', 'B-EQUIP', 'I-YEAR', 'B-TIME', 'I-TIME',
                      'B-DATE', 'I-DATE', 'I-PER', 'B-CONT', 'B-LAW', 'I-LAW', 'I-NORP', 'B-LIT', 'I-LIT', 'B-KIN',
                      'I-ORG', 'I-KIN', 'B-TER', 'B-GOV', 'I-EQUIP', 'B-PRO', 'I-PRO', 'B-WAT', 'I-WAT', 'I-CON',
                      'B-FAC', 'I-FAC', 'B-TAX', 'I-TAX', 'B-LAN', 'I-LAN', 'I-GOV', 'B-SOU', 'I-SOU', 'I-CONT',
                      'I-TER', 'B-SKILL', 'I-SKILL', 'B-IND', 'I-IND', 'B-REL', 'B-SUB', 'B-ANT', 'I-ANT', 'I-REL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # with open('train_dataset_features.txt', 'wb') as fp:
    #     pickle.dump(X_train, fp)
    # with open('train_dataset_labels.txt', 'wb') as fp:
    #     pickle.dump(y_train, fp)
    # with open('test_dataset_features.txt', 'wb') as fp:
    #     pickle.dump(X_test, fp)
    # with open('test_dataset_labels.txt', 'wb') as fp:
    #     pickle.dump(y_test, fp)
    #
    # with open('train_dataset_features.txt', 'rb') as f:
    #     X_train = pickle.load(f)
    # with open('train_dataset_labels.txt', 'rb') as f:
    #     y_train = pickle.load(f)
    # with open('test_dataset_features.txt', 'rb') as f:
    #     X_test = pickle.load(f)
    # with open(the_filename, 'rb') as f:
    #     y_test = pickle.load(f)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=label_category)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=5,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=30,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    filename = '../Model/crf_model.sav'
    # joblib.dump(crf, filename)
    pickle.dump(rs, open(filename, 'wb'))

    # obtaining metrics such as accuracy, etc. on the train set
    labels = list(rs.classes_)
    labels.remove('O')
    print(labels)

    # obtaining metrics such as accuracy, etc. on the test set
    y_pred_test = rs.predict(X_test)
    print('F1 score on the test set = {}\n'.format(flat_f1_score(y_test, y_pred_test,
                                                                 average='weighted', labels=labels)))
    print('Accuracy on the test set = {}\n'.format(flat_accuracy_score(y_test, y_pred_test)))

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print('Test set classification report: \n\n{}'.format(flat_classification_report(y_test, y_pred_test,
                                                                                     labels=sorted_labels, digits=3)))

    # #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    # #stratified_split = StratifiedKFold(n_splits=5)
    #
    # #for train_index, test_index in stratified_split.split(X, y):
    # #    print("TRAIN:", train_index, "TEST:", test_index)
    # #    X_train, X_test = X[train_index], X[test_index]
    # #    y_train, y_test = y[train_index], y[test_index]
    # str_fold = KFold(n_splits=3)
    # # fold = KFold(n_splits=4, random_state=0, shuffle=False)
    #
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     c1=0.1,
    #     c2=0.1,
    #     max_iterations=20,
    #     all_possible_transitions=True,
    # )
    #
    # for train_index, test_index in str_fold.split(X, y):
    #     #print("TRAIN:", train_index, "TEST:", test_index)
    #
    #     #X_train, X_test = X[train_index.astype(int)], X[test_index.astype(int)]
    #     #y_train, y_test = y[int(train_index)], y[int(test_index)]
    #     X_train = [X[index] for index in train_index]
    #     X_test = [X[index] for index in test_index]
    #     y_train = [y[index] for index in train_index]
    #     y_test = [y[index] for index in test_index]
    #
    #
    #     crf.fit(X_train, y_train)
    #
    #     filename = 'crf_model.sav'
    #     # joblib.dump(crf, filename)
    #     pickle.dump(crf, open(filename, 'wb'))
    #
    #     # obtaining metrics such as accuracy, etc. on the train set
    #     labels = list(crf.classes_)
    #     labels.remove('O')
    #     print(labels)
    #
    #     # y_pred_train = crf.predict(X_train)
    #     # print('F1 score on the train set = {}\n'.format(flat_f1_score(y_train, y_pred_train, average='weighted', labels=labels)))
    #     # print('Accuracy on the train set = {}\n'.format(flat_accuracy_score(y_train, y_pred_train)))
    #     #
    #     # sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    #     # print('Train set classification report: \n\n{}'.format(flat_classification_report(y_train, y_pred_train, labels=sorted_labels, digits=3)))
    #
    #     # obtaining metrics such as accuracy, etc. on the test set
    #     y_pred_test = crf.predict(X_test)
    #     print('F1 score on the test set = {}\n'.format(flat_f1_score(y_test, y_pred_test,
    #                                                                  average='weighted', labels=labels)))
    #     print('Accuracy on the test set = {}\n'.format(flat_accuracy_score(y_test, y_pred_test)))
    #
    #     with warnings.catch_warnings():
    #         # ignore all caught warnings
    #         warnings.filterwarnings("ignore")
    #
    #     sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    #     print('Test set classification report: \n\n{}'.format(flat_classification_report(y_test, y_pred_test,
    #                                                                                      labels=sorted_labels, digits=3)))


def convert_to_list():
    sentence_list_string = ""
    sentence_list = []
    tag_list = []

    with open(
            "C:\\Users\ASUS\PycharmProjects\QuestionAnswerGeneration\FrontEnd\dataset\\tagged_dataset\\tagged_data.conll",
            encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            data_tags = csv.reader(line, delimiter='\t')
            tag_list.append(i for i in data_tags)

            if line == '.   PUNCT   O\n':
                sentence_list.append(tag_list)
                sentence_list_string = sentence_list_string + str(tag_list) + ',\n'
                tag_list = []

    return sentence_list_string


def read_conll_as_list():
    with open(
            "C:\\Users\ASUS\PycharmProjects\QuestionAnswerGeneration\FrontEnd\dataset\\tagged_dataset\\tagged_data.conll",
            encoding="utf-8") as file:

        raw_docs = file.read().split('\n\n')
        sentence_list = []
        for doc in raw_docs:
            tags = []
            for line in doc.split('\n'):
                token, pos_tag, ner_tag = line.split('\t')
                tags.append((token, pos_tag, ner_tag))
                print(tags)

            sentence_list.append(tags)
            print(tags)

    return sentence_list
