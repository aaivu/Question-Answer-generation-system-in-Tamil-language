import os

from indicnlp.tokenize import sentence_tokenize
from src.Code.QAG_Module import rulebased
import snowballstemmer
import re

from src.Code.QAG_Module.questiongenerator import ner_question_generation

LIST_FOLDER_PATH = "C:\\Users\ASUS\Desktop\Research Work Upload documents\Github-Aaivu\Question-Answer-generation-system-in-Tamil-language\src\Code\\Utils\lists"

ignore_word_list_file = open(os.path.join(LIST_FOLDER_PATH, "anophara_wordlist.txt"), encoding="utf-8")
ignore_word_list = ignore_word_list_file.read().splitlines()


def get_list_of_sentences(file_content):
    sentences = []
    while file_content != "":
        sentence_regex = re.compile('^[^.]+([0-9]+\.{0,1}\s*){0,}[^.]+[^\s.]{2,}\.')
        match = re.search(sentence_regex, file_content)
        sentences.append(match.string)
        if match:
            file_content = re.sub(sentence_regex, '', file_content)

    return sentences


def check_anaphoric_reference(sentence, first_word, stem_word):
    person_refs = ['அவர்', 'அவருக்கு', 'அவரை', 'அவரின்', 'அவரது', 'அவரால்', 'இவர்', 'இவரை', 'இவரது',
                   'இவரின்', 'இவருக்கு', 'அவரால்']
    location_refs = ['அங்கே', 'இங்கே', 'அங்கு', 'இங்கு']

    return True


def check_if_sentence_suitable(sentence):
    first_word = sentence.partition(' ')[0]
    if first_word in ignore_word_list:
        return False
    stemmer = snowballstemmer.stemmer('tamil')
    stem_word = stemmer.stemWords(first_word.split())
    if stem_word and stem_word[0] in ignore_word_list:
        return False

    return check_anaphoric_reference(sentence, first_word, stem_word)


def process_file(file_content, file_write_path):
    stop_words_file = open(os.path.join(LIST_FOLDER_PATH, "stopwords.txt"), encoding="utf-8")
    stop_words = stop_words_file.read().splitlines()

    writefile = open(file_write_path, "w", encoding="utf-8")
    sentences = sentence_tokenize.sentence_split(file_content, lang='tam')

    cleaned_sentences = []
    writefile.write("Questions and Answers generated from Rule based Module" + "\r\n")
    for sentence in sentences:
        sentence = ' '.join(sentence.split())
        sentence = re.sub('\u200c', '', sentence)
        print(sentence)
        is_sentence_suitable = check_if_sentence_suitable(sentence)
        if not is_sentence_suitable:
            continue

        sentence = re.sub(r'\([^)]*\)', '', sentence)
        cleaned_sentences.append(sentence)
        rulebased.regex_match_date_time(sentence, writefile)
        rulebased.regex_match_quantity(sentence, writefile)
        rulebased.process_single_quote(sentence, writefile)
        rulebased.regex_match_multiple_items(sentence, writefile)

        rulebased.check_gazetteer(sentence, writefile)

    writefile.write("Questions and Answers generated from NER Module" + "\r\n")
    ner_question_generation(cleaned_sentences, writefile)

