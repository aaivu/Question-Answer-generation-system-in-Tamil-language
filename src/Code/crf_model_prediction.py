import pickle
import re

from rippletagger.tagger import Tagger
from dataset.tagging_utils import sent2features


def load_model():
    #loaded_model = joblib.load('crf_model.sav')
    loaded_model = pickle.load(open('crf_model.sav', 'rb'))
    return loaded_model


def tag_sentence(sentences):
    sentence_tag_list = []
    for sentence in sentences:
        sentence = ' '.join(sentence.split())
        sentence = re.sub('\u200c', '', sentence)
        tagger = Tagger(language='tam')
        pos_tagger = tagger.tag(sentence)
        sentence_tag = []
        for (word, tag) in pos_tagger:
            sentence_tag.append((word, tag, 'O'))

        sentence_tag_list.append(sentence_tag)
    return sentence_tag_list


def predict_ner_tag(sentences):
    tagged_data = tag_sentence(sentences)
    print(tagged_data)
    features = [sent2features(ts) for ts in tagged_data]
    predicted = load_model().predict(features)

    print(predicted)
    return predicted
