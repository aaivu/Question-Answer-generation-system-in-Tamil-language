import pickle
import warnings

import scipy
import sklearn_crfsuite
import snowballstemmer
import csv

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite.metrics import flat_accuracy_score
from sklearn_crfsuite.metrics import flat_f1_score

from FrontEnd import gazetteers
import re

from FrontEnd.tamilutils import findinflectionsuffix

label_category = ['B-NAME', 'I-NAME', 'B-YEAR', 'B-MON', 'I-MON', 'B-COU', 'I-COU', 'B-COM', 'I-COM', 'B-CITY',
                      'I-CITY', 'B-NORP', 'B-EVE', 'I-EVE', 'B-LOC', 'I-LOC', 'B-TRO', 'I-TRO', 'B-NUM', 'B-PER',
                      'B-CON', 'B-ORG', 'B-SEC', 'I-SEC', 'B-DES', 'I-DES', 'B-EQUIP', 'I-YEAR', 'B-TIME', 'I-TIME',
                      'B-DATE', 'I-DATE', 'I-PER', 'B-CONT', 'B-LAW', 'I-LAW', 'I-NORP', 'B-LIT', 'I-LIT', 'B-KIN',
                      'I-ORG', 'I-KIN', 'B-TER', 'B-GOV', 'I-EQUIP', 'B-PRO', 'I-PRO', 'B-WAT', 'I-WAT', 'I-CON',
                      'B-FAC', 'I-FAC', 'B-TAX', 'I-TAX', 'B-LAN', 'I-LAN', 'I-GOV', 'B-SOU', 'I-SOU', 'I-CONT',
                      'I-TER', 'B-SKILL', 'I-SKILL', 'B-IND', 'I-IND', 'B-REL', 'B-SUB', 'B-ANT', 'I-ANT', 'I-REL']

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=label_category)

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

    filename = 'crf_model.sav'
    pickle.dump(rs, open(filename, 'wb'))

    labels = list(rs.classes_)
    labels.remove('O')

    y_pred_test = rs.predict(X_test)
    print('F1 score on the test set = {}\n'.format(flat_f1_score(y_test, y_pred_test,
                                                                 average='weighted', labels=labels)))
    print('Accuracy on the test set = {}\n'.format(flat_accuracy_score(y_test, y_pred_test)))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print('Test set classification report: \n\n{}'.format(flat_classification_report(y_test, y_pred_test,
                                                                                     labels=sorted_labels, digits=3)))

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



























rules = {
    "COU": {
        "ஐ": "எந்த நாட்டை",
        "ஆல்": "எந்த நாட்டால்",
        "கு": "எந்த நாட்டுக்கு",
        "இல்": "எந்த நாட்டில்",
        "இன்": "எந்த நாட்டின்",
        "அது": "எந்த நாட்டினது",
        "உம்": "எந்த நாடும்",
        "மீது": "எந்த நாட்டின்மீது",
        "உடன்": "எந்த நாட்டுடன்",
        "லிருந்து": "எந்த நாட்டிலிருந்து",
        "plural": "எந்த நாடு",
        "mei": "எந்த நாட்டு",
        "None": "எந்த நாடு"
    },

    "CITY": {
        "தலைநகர்": {
            "ஐ": "எந்த தலைநகரத்தை",
            "ஆல்": "எந்த தலைநகரத்தால்",
            "கு": "எந்த தலைநகரத்துக்கு",
            "இல்": "எந்த தலைநகரத்தில்",
            "இன்": "எந்த தலைநகரத்தின்",
            "அது": "எந்த தலைநகரத்தினது",
            "உம்": "எந்த தலைநகரமும்",
            "மீது": "எந்த தலைநகரத்தின்மீது",
            "உடன்": "எந்த தலைநகரத்துடன்",
            "லிருந்து": "எந்த தலைநகரத்திலிருந்து",
            "plural": "எந்த தலைநகரங்கள்",
            "mei": "எந்த தலைநகர",
            "None": "எந்த தலைநகரம்"
        },
        "ஐ": "எந்த நகரத்தை",
        "ஆல்": "எந்த நகரத்தால்",
        "கு": "எந்த நகரத்துக்கு",
        "இல்": "எந்த நகரத்தில்",
        "இன்": "எந்த நகரத்தின்",
        "அது": "எந்த நகரத்தினது",
        "உம்": "எந்த நகரமும்",
        "மீது": "எந்த நகரத்தின்மீது",
        "உடன்": "எந்த நகரத்துடன்",
        "லிருந்து": "எந்த நகரத்திலிருந்து",
        "plural": "எந்த நகரங்கள்",
        "mei": "எந்த நகர",
        "None": "எந்த நகரம்"
    },

    "CONT": {
        "ஐ": "எந்த கண்டத்தை",
        "ஆல்": "எந்த கண்டத்தால்",
        "கு": "எந்த கண்டத்துக்கு",
        "இல்": "எந்த கண்டத்தில்",
        "இன்": "எந்த கண்டத்தின்",
        "அது": "எந்த கண்டதஂதினது",
        "உம்": "எந்த கண்டமும்",
        "மீது": "எந்த கண்டத்தின்மீது",
        "உடன்": "எந்த கண்டதஂதுடன்",
        "லிருந்து": "எந்த கண்டத்திலிருந்து",
        "plural": "எந்த கண்டங்கள்",
        "mei": "எந்த கண்டத்து",
        "None": "எந்த கண்டம்"
    },

    "KIN": {
        "ஐ": "எந்த இராச்சியத்தை",
        "ஆல்": "எந்த இராச்சியத்தால்",
        "கு": "எந்த இராச்சியத்துக்கு",
        "இல்": "எந்த இராச்சியத்தில்",
        "இன்": "எந்த இராச்சியத்தின்",
        "அது": "எந்த இராச்சியத்தினது",
        "உம்": "எந்த இராச்சியமும்",
        "மீது": "எந்த இராச்சியத்தின்மீது",
        "உடன்": "எந்த இராச்சியத்துடன்",
        "லிருந்து": "எந்த இராச்சியத்திலிருந்து",
        "plural": "எந்த இராச்சியங்கள்",
        "mei": "எந்த இராச்சிய",
        "None": "எந்த இராச்சியம்"
    },

    "PER": {
        "மன்னன்": {
            "ஐ": "எந்த மன்னனை ",
            "ஆல்": "எந்த மன்னனால்",
            "கு": "எந்த மன்னனுக்கு",
            "இன்": "எந்த மன்னனின்",
            "அது": "எந்த மன்னனது",
            "உம்": "எந்த மன்னனும்",
            "மீது": "எந்த மன்னன்மீது",
            "உடன்": "எந்த மன்னனுடன்",
            "plural": "எந்த மன்னர்கள்",
            "None": "எந்த மன்னன்"
        },
        "ஜனாதிபதி": {
            "ஐ": "எந்த ஜனாதிபதியை",
            "ஆல்": "எந்த ஜனாதிபதியால்",
            "கு": "எந்த ஜனாதிபதிக்கு",
            "இன்": "எந்த ஜனாதிபதியின்",
            "அது": "எந்த ஜனாதிபதியினது",
            "உம்": "எந்த ஜனாதிபதியும்",
            "மீது": "எந்த ஜனாதிபதிமீது",
            "உடன்": "எந்த ஜனாதிபதியுடன்",
            "plural": "எந்த ஜனாதிபதிகள்",
            "None": "எந்த ஜனாதிபதி"
        },
        "ஐ": "யாரை",
        "ஆல்": "யாரால்",
        "கு": "யாருக்கு",
        "இன்": "யாரின்",
        "அது": "யாரது",
        "உம்": "யாரும்",
        "மீது": "யாரின்மீது",
        "உடன்": "யாருடன்",
        "None": "யார்"
    },

    "EVE": {
        "ஐ": "எந்த நிகழ்வை",
        "ஆல்": "எந்த நிகழ்வால்",
        "கு": "எந்த நிகழ்வுக்கு",
        "இல்": "எந்த நிகழ்வில்",
        "இன்": "எந்த நிகழ்வின்",
        "அது": "எந்த நிகழ்வினது",
        "உம்": "எந்த நிகழ்வும்",
        "உடன்": "எந்த நிகழ்வுடன்",
        "mei": "எந்த நிகழ்வு",
        "plural": "எந்த நிகழ்வுகள்",
        "None": "எந்த நிகழ்வு"
    },

    "MON": {
        "இல்": "எந்த மாதத்தில்",
        "இன்": "எந்த மாதத்தின்",
        "plural": "எந்த மாதங்கள்",
        "None": "எந்த மாதம்"
    },

    "YEAR": {
        "இல்": "எந்த வருடத்தில்",
        "இன்": "எந்த வருடத்தின்",
        "None": "எத்தனையாம் ஆண்டு"
    },
    "TRO": {
        "படை": {
            "ஐ": "எந்த படையை",
            "ஆல்": "எந்த படையால்",
            "கு": "எந்த படைக்கு",
            "இல்": "எந்த படையில்",
            "இன்": "எந்த படையின்",
            "அது": "எந்த படையினது",
            "உம்": "எந்த படையும்",
            "மீது": "எந்த படையின்மீது",
            "உடன்": "எந்த படையுடன்",
            "லிருந்து": "எந்த படையிலிருந்து",
            "mei": "எந்த படை",
            "None": "எந்த படை"
        },
        "படையினர்": {

        },
        "இராணுவம்": {
            "ஐ": "எந்த இராணுவத்தை",
            "ஆல்": "எந்த இராணுவத்தால்",
            "கு": "எந்த இராணுவத்துக்கு",
            "இல்": "எந்த இராணுவத்தில்",
            "இன்": "எந்த இராணுவத்தின்",
            "அது": "எந்த இராணுவத்தினது",
            "உம்": "எந்த இராணுவமும்",
            "மீது": "எந்த இராணுவத்தின்மீது",
            "உடன்": "எந்த இராணுவத்துடன்",
            "லிருந்து": "எந்த இராணுவத்திலிருந்து",
            "ஆகிய": "எந்த இராணுவம்",
            "None": "எந்த இராணுவம்"
        },
        "இயக்கம்": {
            "ஐ": "எந்த இயக்கத்தை",
            "ஆல்": "எந்த இயக்கத்தால்",
            "கு": "எந்த இயக்கத்துக்கு",
            "இல்": "எந்த இயக்கத்தில்",
            "இன்": "எந்த இயக்கத்தின்",
            "அது": "எந்த இயக்கத்தினது",
            "உம்": "எந்த இயக்கமும்",
            "மீது": "எந்த இயக்கத்தின்மீது",
            "உடன்": "எந்த இயக்கத்துடன்",
            "லிருந்து": "எந்த இயக்கத்திலிருந்து",
            "ஆகிய": "எந்த இயக்கம்",
            "None": "எந்த இயக்கம்"
        },

    },

    "ORG": {
        "கட்சி": {
            "ஐ": "எந்த கட்சியை ",
            "ஆல்": "எந்த கட்சியால்",
            "கு": "எந்த கட்சிக்கு",
            "இல்": "எந்த கட்சியில்",
            "இன்": "எந்த கட்சியின்",
            "அது": "எந்த கட்சியினது",
            "உம்": "எந்த கட்சியும்",
            "மீது": "எந்த கட்சியின்மீது",
            "உடன்": "எந்த கட்சியுடன்",
            "லிருந்து": "எந்த கட்சியிலிருந்து",
            "ஆகிய": "எந்த கட்சி",
            "None": "எந்த கட்சி"
        },
        "சபை": {
            "ஐ": "எந்த சபையை",
            "ஆல்": "எந்த சபையால்",
            "கு": "எந்த சபைக்கு",
            "இல்": "எந்த சபையில்",
            "இன்": "எந்த சபையின்",
            "அது": "எந்த சபையினது",
            "உம்": "எந்த சபையும்",
            "மீது": "எந்த சபையின்மீது",
            "உடன்": "எந்த சபையுடன்",
            "லிருந்து": "எந்த சபையிலிருந்து",
            "ஆகிய": "எந்த சபை",
            "None": "எந்த சபை"
        }
    },
    'CON': {
        'கொள்கை': {

        },
        "சிந்தனை": {

        },
        "எண்ணக்கரு": {

        },
        "முறை": {

        },
        "கருத்து": {

        },

    },
    'CUR': {
        "ஐ": "எந்த சபையை",
        "ஆல்": "எந்த சபையால்",
        "கு": "எந்த சபைக்கு",
        "இல்": "எந்த சபையில்",
        "இன்": "எந்த சபையின்",
        "அது": "எந்த சபையினது",
        "உம்": "எந்த சபையும்",
        "மீது": "எந்த சபையின்மீது",
        "உடன்": "எந்த சபையுடன்",
        "லிருந்து": "எந்த சபையிலிருந்து",
        "ஆகிய": "எந்த சபை",
        "None": "எந்த சபை"
    },
    'GOD': {
        "ஐ": "எந்த சபையை",
        "ஆல்": "எந்த சபையால்",
        "கு": "எந்த சபைக்கு",
        "இல்": "எந்த சபையில்",
        "இன்": "எந்த சபையின்",
        "அது": "எந்த சபையினது",
        "உம்": "எந்த சபையும்",
        "மீது": "எந்த சபையின்மீது",
        "உடன்": "எந்த சபையுடன்",
        "லிருந்து": "எந்த சபையிலிருந்து",
        "ஆகிய": "எந்த சபை",
        "None": "எந்த சபை"
    },
    'IND': {
        "ஐ": "எந்த கைத்தொழிலை",
        "ஆல்": "எந்த கைத்தொழிலால்",
        "கு": "எந்த கைத்தொழிலுக்கு",
        "இல்": "எந்த கைத்தொழிலில்",
        "இன்": "எந்த கைத்தொழிலின்",
        "அது": "எந்த கைத்தொழிலிலினது",
        "உம்": "எந்த கைத்தொழிலும்",
        "மீது": "எந்த கைத்தொழில்மீது",
        "உடன்": "எந்த கைத்தொழிலுடன்",
        "லிருந்து": "எந்த கைத்தொழிலிருந்து",
        "ஆகிய": "எந்த கைத்தொழில்",
        "None": "எந்த கைத்தொழில்"
    },
    'SEC': {
        'துறை'
        "ஐ": "எந்த துறையை",
        "ஆல்": "எந்த துறையால்",
        "கு": "எந்த துறைக்கு",
        "இல்": "எந்த துறையில்",
        "இன்": "எந்த துறையின்",
        "அது": "எந்த துறையினது",
        "உம்": "எந்த துறையும்",
        "மீது": "எந்த துறையின்மீது",
        "உடன்": "எந்த துறையுடன்",
        "லிருந்து": "எந்த துறையிலிருந்து",
        "ஆகிய": "எந்த துறை",
        "None": "எந்த துறை"
    },
    'SKILL': ['கலை', 'அறிவு', 'கலையம்சம்', 'கலையம்சங்கள்'],
    'SOU': {
        'மூலாதாரம்': {
        },
        'மூலாதாரங்கள்': {

        },
    },
    'SUB': ['பாடம்', 'பாடங்கள்'],
    'TER': ['பிரதேசம்', 'ஊர்', 'தீவு', 'குடியேற்றம்'],
    'LAN': {
        "ஐ": "எந்த மொழியை",
        "ஆல்": "எந்த மொழியால்",
        "கு": "எந்த மொழிக்கு",
        "இல்": "எந்த மொழியில்",
        "இன்": "எந்த மொழியின்",
        "அது": "எந்த மொழியினது",
        "உம்": "எந்த மொழியும்",
        "உடன்": "எந்த மொழியுடன்",
        "லிருந்து": "எந்த மொழியிலிருந்து",
        "ஆகிய": "எந்த மொழி",
        "None": "எந்த மொழி"
    },
    'LAW': {
        'சட்டம்': {

        },
        'சீர்த்திருத்தம்': {

        },
        'பிரகடனம்': {

        },
        'கொள்கை': {

        },
        'ஒப்பந்தம்': {

        }
    },
    'LIT': {
        'இலக்கியம்': {

        },
        'காவியம்': {

        },

    },
    'LOC': {
        'மாளிகை': {

        },
        'விகாரை': {

        },
        'கோயில்': {

        },
        'துறைமுகம்': {

        },
        'பாராளுமன்றம்': {

        },
        'முகாம்': {
        },
        'சிறைச்சாலை': {

        }
    },
    'DES': ['பதவி'],
    'PRO': {
        'பொருட்கள்': {
            "ஐ": "எந்த மொழியை",
            "ஆல்": "எந்த மொழியால்",
            "கு": "எந்த மொழிக்கு",
            "இல்": "எந்த மொழியில்",
            "இன்": "எந்த மொழியின்",
            "அது": "எந்த மொழியினது",
            "உம்": "எந்த மொழியும்",
            "உடன்": "எந்த மொழியுடன்",
            "லிருந்து": "எந்த மொழியிலிருந்து",
            "ஆகிய": "எந்த மொழி",
            "None": "எந்த மொழி"
        },
        "ஐ": "எந்த பொருளை",
        "ஆல்": "எந்த பொருளால்",
        "கு": "எந்த பொருளுக்கு",
        "இல்": "எந்த பொருளில்",
        "இன்": "எந்த பொருளின்",
        "அது": "எந்த பொருளினது",
        "உம்": "எந்த பொருளும்",
        "உடன்": "எந்த பொருளுடன்",
        "லிருந்து": "எந்த பொருளிலிருந்து",
        "ஆகிய": "எந்த பொருள்",
        "None": "எந்த பொருள்"

    },
    'REL': {
        "ஐ": "எந்த சமயத்தை",
        "ஆல்": "எந்த சமயத்தால்",
        "கு": "எந்த சமயத்துக்கு",
        "இல்": "எந்த சமயத்தில்",
        "இன்": "எந்த சமயத்தின்",
        "அது": "எந்த சமயத்தினது",
        "உம்": "எந்த சமயமும்",
        "மீது": "எந்த சமயத்தின்மீது",
        "லிருந்து": "எந்த சமயத்திலிருந்து",
        "ஆகிய": "எந்த சமயம்",
        "None": "எந்த சமயம்"
    },

    'NORP': {
        "வம்சத்தினர்": {
            "ஐ": "எந்த வம்சத்தினரை",
            "ஆல்": "எந்த வம்சத்தினரால்",
            "கு": "எந்த வம்சத்தினருக்கு",
            "இல்": "எந்த வம்சத்தினரில்",
            "இன்": "எந்த வம்சத்தினரின்",
            "அது": "எந்த வம்சத்தினரது",
            "உம்": "எந்த வம்சத்தினரும்",
            "மீது": "எந்த வம்சத்தினரின்மீது",
            "ஆகிய": "எந்த வம்சத்தினர்",
            "None": "எந்த வம்சத்தினர்"
        },
        'வம்சம்': {
            "ஐ": "எந்த வம்சத்தை",
            "ஆல்": "எந்த வம்சத்தால்",
            "கு": "எந்த வம்சத்துக்கு",
            "இல்": "எந்த வம்சத்தில்",
            "இன்": "எந்த வம்சத்தின்",
            "அது": "எந்த வம்சத்தினது",
            "உம்": "எந்த வம்சமும்",
            "மீது": "எந்த வம்சத்தின்மீது",
            "ஆகிய": "எந்த வம்சம்",
            "None": "எந்த வம்சம்"
        },
        'வகுப்பினர்': {
            "ஐ": "எந்த வகுப்பினரை",
            "ஆல்": "எந்த வகுப்பினரால்",
            "கு": "எந்த வகுப்பினருக்கு",
            "இல்": "எந்த வகுப்பினரில்",
            "இன்": "எந்த வகுப்பினரின்",
            "அது": "எந்த வகுப்பினரது",
            "உம்": "எந்த வகுப்பினரும்",
            "மீது": "எந்த வகுப்பினர்மீது",
            "ஆகிய": "எந்த வகுப்பினர்",
            "None": "எந்த வகுப்பினர்"
        },
        'இனத்தவர்': {
            "ஐ": "எந்த இனத்தவரை",
            "ஆல்": "எந்த இனத்தவரால்",
            "கு": "எந்த இனத்தவருக்கு",
            "இல்": "எந்த இனத்தவரில்",
            "இன்": "எந்த இனத்தவரின்",
            "அது": "எந்த இனத்தவரது",
            "உம்": "எந்த இனத்தவரும்",
            "மீது": "எந்த இனத்தவரின்மீது",
            "ஆகிய": "எந்த இனத்தவர்",
            "None": "எந்த இனத்தவர்"
        },
        'சமுதாயம்': {
            "ஐ": "எந்த சமுதாயத்தை",
            "ஆல்": "எந்த சமுதாயத்தால்",
            "கு": "எந்த சமுதாயத்துக்கு",
            "இல்": "எந்த சமுதாயத்தில்",
            "இன்": "எந்த சமுதாயத்தின்",
            "அது": "எந்த சமுதாயத்தினது",
            "உம்": "எந்த சமுதாயமும்",
            "மீது": "எந்த சமுதாயத்தின்மீது",
            "ஆகிய": "எந்த சமுதாயம்",
            "None": "எந்த சமுதாயம்"
        },
        'சமுதாயத்தினர்': {
            "ஐ": "எந்த சமுதாயத்தினரை",
            "ஆல்": "எந்த சமுதாயத்தினரால்",
            "கு": "எந்த சமுதாயத்தினருக்கு",
            "இல்": "எந்த சமுதாயத்தினரில்",
            "இன்": "எந்த சமுதாயத்தினரின்",
            "அது": "எந்த சமுதாயத்தினரது",
            "உம்": "எந்த சமுதாயத்தினரும்",
            "மீது": "எந்த சமுதாயத்தினரின்மீது",
            "ஆகிய": "எந்த சமுதாயத்தினர்",
            "None": "எந்த சமுதாயத்தினர்"
        },
            "ஐ": "எந்த வம்சத்தினரை",
            "ஆல்": "எந்த வம்சத்தினரால்",
            "கு": "எந்த வம்சத்தினருக்கு",
            "இல்": "எந்த வம்சத்தினரில்",
            "இன்": "எந்த வம்சத்தினரின்",
            "அது": "எந்த வம்சத்தினரது",
            "உம்": "எந்த வம்சத்தினரும்",
            "மீது": "எந்த வம்சத்தினரின்மீது",
            "ஆகிய": "எந்த வம்சத்தினர்",
            "None": "எந்த வம்சத்தினர்"
    },
    'GOV': {
        "ஐ": "எந்த அரசை",
        "ஆல்": "எந்த அரசால்",
        "கு": "எந்த அரசுக்கு",
        "இல்": "எந்த அரசில்",
        "இன்": "எந்த அரசின்",
        "அது": "எந்த அரசினது",
        "உம்": "எந்த அரசும்",
        "மீது": "எந்த அரசின்மீது",
        "லிருந்து": "எந்த அரசிலிருந்து",
        "ஆகிய": "எந்த அரசு",
        "None": "எந்த அரசு"
    },
}