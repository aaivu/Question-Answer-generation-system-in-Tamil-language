import pickle
import warnings

import scipy
import sklearn_crfsuite
import snowballstemmer
import re

from rippletagger.tagger import Tagger
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics
from FrontEnd import gazetteers

from FrontEnd import training_data, testing_data

def lookup_search(lookup):
    for key, value in gazetteers.gazetteer_list.items():
        for v in value:
            partnames = v.split(" ")
            if lookup in partnames:
                return key
    return 'None'

def isyearformat(word):
    return (re.match('^[1-2][0-9]{3}(?:இல்|ல்|ஆம்|ம்){0,1}', word) != 'None')

def checkifordinalnumericword(word):
    return word in gazetteers.previous_word.get('ORDINAL')

def checkprefixword(word):
    if word in gazetteers.previous_word.get('LOC'):
        return 'location'
    if word in gazetteers.previous_word.get('COU'):
        return 'country'
    if word in gazetteers.previous_word.get('ORG'):
        return 'org'
    if word in gazetteers.previous_word.get('TIME'):
        return 'time'

    return 'None'

def checkifwordendswithcomma(word):
    return word.endswith(',')

def wordisaconjunction(word):
    common_conj_words = ['மற்றும்', 'ஆகிய',  'ஆகியன', 'போன்ற', 'போன்றன', 'போன்றவை', 'ஆகியவை',
                         'என்பன', 'என்பவை', 'ஆகியோர்', 'போன்றோர்']
    return word in common_conj_words

def checkifwordisnumber(word):
    if word.isdigit():
        return True
    for key, value in gazetteers.numbers.items():
        for v in value:
            if word == v:
                return True
    return False

def checkcluewordlist(word, key):
    stemmer = snowballstemmer.stemmer('tamil')
    stemword = stemmer.stemWords(word.split())
    if word in gazetteers.clue_words.get(key) or stemword in gazetteers.clue_words.get(key):
        return True

    return False

def getstemword(word):
    stemmer = snowballstemmer.stemmer('tamil')
    return stemmer.stemWords(word.split())

def iswordcontainshyphen(word):
    return re.search(r'-', word) != 'None'

def ishyphen(word):
    return word == '-'

def isclueword(word):
    if checkcluewordlist(word, 'CITY'):
        return 'city'

    elif checkcluewordlist(word, 'COU'):
        return 'country'

    elif checkcluewordlist(word, 'CON'):
        return 'continent'

    elif checkcluewordlist(word, 'PER'):
        return 'person'

    elif checkcluewordlist(word, 'ORG'):
        return 'org'

    elif checkcluewordlist(word, 'TIME'):
        return 'time'

    elif checkcluewordlist(word, 'NUM'):
        return 'quantity'

    elif checkcluewordlist(word, 'TRO'):
        return 'troop'

    elif checkcluewordlist(word, 'EVE'):
        return 'event'

    elif checkcluewordlist(word, 'GPE'):
        return 'gpe'

    elif checkcluewordlist(word, 'GOV'):
        return 'gov'

    else:
        return 'None'

def gazettercheck(word):
    return lookup_search(word)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        # 'word[-2:]': word[-2:],
        'isnumber': checkifwordisnumber(word),
        'postag': postag,
        'clueword': isclueword(word),
        'gazetteerword': gazettercheck(word),
        'isordinal': checkifordinalnumericword(word),
        'isyearformat': isyearformat(word),
        'endwithcomma': checkifwordendswithcomma(word),
        'containhyphen': iswordcontainshyphen(word),
        'ishyphen': ishyphen(word),
        'stemword': getstemword(word)
    }

    if i > 0:
        pword1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word': pword1,
            '-1:postag': postag1,
            '-1:clueword': isclueword(pword1),
            '-1:isnumber': checkifwordisnumber(pword1),
            '-1:gazetteerword': gazettercheck(pword1),
            '-1:isordinal': checkifordinalnumericword(pword1),
            '-1:isyearformat': isyearformat(pword1),
            '-1:isconjunction': wordisaconjunction(pword1),
            '-1:endwithcomma': checkifwordendswithcomma(pword1),
            # '-1:prefixword': checkprefixword(pword1),
            '-1:stemword': getstemword(pword1),
            '-1:ishyphen': ishyphen(word),
        })
        if i > 1:
            pword2 = sent[i-2][0]
            postag2 = sent[i-2][1]
            features.update({
                '-2:word': pword2,
                '-2:postag': postag2,
                '-2:gazetteerword': gazettercheck(pword2),
                '-2:isyearformat': isyearformat(pword2),
                '-2:clueword': isclueword(pword2),
                '-2:endwithcomma': checkifwordendswithcomma(pword2)
            })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        nword1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word': nword1,
            '+1:postag': postag1,
            '+1:clueword': isclueword(nword1),
            '+1:gazetteerword': gazettercheck(nword1),
            '+1:isnumber': checkifwordisnumber(nword1),
            '+1:isconjunction': wordisaconjunction(nword1),
            '+1:endwithcomma': checkifwordendswithcomma(nword1),
            '+1:ishyphen': ishyphen(word),
            '+1:stemword': getstemword(nword1)
        })
        if i < len(sent) - 2:
            nword2 = sent[i+2][0]
            postag2 = sent[i+2][1]
            features.update({
                '+2:word': nword2,
                '+2:postag': postag2,
                '+2:gazetteerword': gazettercheck(nword2),
                '+2:clueword': isclueword(nword2),
                '+2:isconjunction': wordisaconjunction(nword2),
                '+2:endwithcomma': checkifwordendswithcomma(nword2)
            })

    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def trainandtest():
    train_sents = training_data.get_training_data()
    test_sents = testing_data.get_testing_data()

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=20,
            all_possible_transitions=True,
        )
    crf.fit(X_test, y_test)
    # crf = sklearn_crfsuite.CRF(
    #     algorithm='lbfgs',
    #     max_iterations=100,
    #     all_possible_transitions=True
    # )
    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.05),
    # }
    # crf.fit(X_train, y_train)

    # obtaining metrics such as accuracy, etc. on the train set
    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)

    # print('F1 score on the train set without hyper tuning = {}\n'.format(
    #     metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)))


    # use the same metric for evaluation
    # f1_scorer = make_scorer(metrics.flat_f1_score,
    #                         average='weighted', labels=labels)

    # search
    # rs = RandomizedSearchCV(crf, params_space,
    #                         cv=3,
    #                         verbose=1,
    #                         n_jobs=-1,
    #                         n_iter=50,
    #                         scoring=f1_scorer)
    # rs.fit(X_train, y_train)

    filename = 'crf_model.sav'
    #joblib.dump(crf, filename)
    pickle.dump(crf, open(filename, 'wb'))

    ypred = crf.predict(X_train)
    print('F1 score on the train set = {}\n'.format(
        metrics.flat_f1_score(y_train, ypred, average='weighted', labels=labels)))
    print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(y_train, ypred)))

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print('Train set classification report: \n\n{}'.format(metrics.flat_classification_report(
        y_train, ypred, labels=sorted_labels, digits=3
    )))
    # obtaining metrics such as accuracy, etc. on the test set
    ypred = crf.predict(X_test)
    print('F1 score on the test set = {}\n'.format(metrics.flat_f1_score(y_test, ypred,
                                                                         average='weighted', labels=labels)))
    print('Accuracy on the test set = {}\n'.format(metrics.flat_accuracy_score(y_test, ypred)))

    # sorted_labels = sorted(
    #     labels,
    #     key=lambda name: (name[1:], name[0])
    # )
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")

    # print('Test set classification report: \n\n{}'.format(metrics.flat_classification_report(
    #     y_test, ypred,labels=sorted_labels, digits=3)))

def tagsentence(sentence):
    tagger = Tagger(language='tam')
    postagger = tagger.tag(sentence)
    sentencetaglist = []
    for (word, tag) in postagger:
        sentencetaglist.append((word, tag, 'O'))

    return sentencetaglist
