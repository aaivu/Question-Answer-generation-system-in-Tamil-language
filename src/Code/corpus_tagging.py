import re
from rippletagger.tagger import Tagger
from indicnlp.tokenize import sentence_tokenize


def ner_tagging():
    ner_tagging_file = open('files/training_data_renaissance.txt', encoding="utf-8")
    ner_tagging_sentences = ner_tagging_file.read()
    sentences = sentence_tokenize.sentence_split(ner_tagging_sentences, lang='tam')

    writefile = open("ner_training_dataset.txt", mode="a", encoding="utf-8")
    ner_sentences_tag = []
    ner_sentences_tag_string = ""
    for sentence in sentences:
        sentence = ' '.join(sentence.split())
        sentence = re.sub('\u200c', '', sentence)
        tagger = Tagger(language='tam')
        pos_tagger = tagger.tag(sentence)
        ner_tag_list = []
        for (word, tag) in pos_tagger:
            ner_tag_list.append((word, tag, 'O'))

        ner_sentences_tag.append(ner_tag_list)
        ner_sentences_tag_string = ner_sentences_tag_string + str(ner_tag_list) + ',\n'

    writefile.write(ner_sentences_tag_string)
    writefile.close()

    ner_tagging_file = open('files/testing_data_renaissance.txt', encoding="utf-8")
    ner_tagging_sentences = ner_tagging_file.read()
    sentences = sentence_tokenize.sentence_split(ner_tagging_sentences, lang='tam')

    writefile = open("ner_testing_dataset.txt", mode="a", encoding="utf-8")
    ner_sentences_tag = []
    ner_sentences_tag_string = ""
    for sentence in sentences:
        sentence = ' '.join(sentence.split())
        sentence = re.sub('\u200c', '', sentence)
        tagger = Tagger(language='tam')
        pos_tagger = tagger.tag(sentence)
        ner_tag_list = []
        for (word, tag) in pos_tagger:
            ner_tag_list.append((word, tag, 'O'))

        ner_sentences_tag.append(ner_tag_list)
        ner_sentences_tag_string = ner_sentences_tag_string + str(ner_tag_list) + ',\n'

    writefile.write(ner_sentences_tag_string)
    writefile.close()
