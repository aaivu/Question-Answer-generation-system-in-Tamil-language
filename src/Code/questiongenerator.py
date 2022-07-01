import snowballstemmer
import tamil

import tamilutils, gazetteers, grammaticalrules
from crf_model_prediction import predict_ner_tag
from dataset.tagging_utils import is_year_format, check_if_word_is_number
from run import write_qa_file, write_gap_fill_question

mei_letters = [u"க்", u"ச்", u"ட்", u"த்", u"ப்", u"ற்",
               u"ஞ்", u"ங்", u"ண்", u"ந்", u"ம்", u"ன்",
               u"ய்", u"ர்", u"ல்", u"வ்", u"ழ்", u"ள்"]


def return_inflection_suffix(word):
    letters = tamil.utf8.get_letters(word)
    if len(letters) > 1:
        last_letter = letters[-1]
        second_last_letter = letters[-2]
        last_letter_unicode = tamilutils.getUnicodePoints(last_letter)
        third_last_letter = letters[-3] if len(letters) > 2 else ''
        forth_last_letter = letters[-4] if len(letters) > 3 else ''

        if last_letter == u"கு":
            if second_last_letter == u"க்" or second_last_letter == u"ற்":
                return u"கு"
        elif last_letter == u"ன்":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"இ":
                return u"இன்"
        elif last_letter == u"ல்":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"ஆ":
                return u"ஆல்"
            elif tuple[1] == u"இ":
                return u"இல்"
        elif last_letter == u"து":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"அ":
                return u"அது"
        elif word.endswith(u"லும்"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"இ":
                    return u"இலும்"
        elif last_letter == u"ம்":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"உ":
                return u"உம்"
        elif last_letter == u"ன":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"ஆ":
                return u"ஆன"
        elif last_letter_unicode[1] == u"ஐ":
            return u"ஐ"
        elif word.endswith(u"டைய"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"உ":
                    return u"உடைய"
        elif word.endswith(u"டன்"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"உ":
                    return u"உடன்"
        elif word.endswith(u"டம்"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"இ":
                    return u"இடம்"
        elif word.endswith(u"கிய"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"ஆ":
                    return u"ஆகிய"
        elif word.endswith(u"ருந்து"):
            if forth_last_letter != '':
                tuple = tamilutils.getUnicodePoints(forth_last_letter)
                if tuple[1] == u"இ":
                    return u"இருந்து"
        elif word.endswith('மீது'):
            return u"மீது"

        return 'None'
    return 'None'


def question_generation(sentence, word, prev_word, next_word, named_entity_type, writefile):
    list_of_words = word.split()
    last_entity_word = list_of_words[-1]
    inflection_suffix = return_inflection_suffix(last_entity_word)
    if prev_word != '' and search_clue_word_list(prev_word, named_entity_type) != 'None':
        question_word = search_rules(named_entity_type, inflection_suffix)
        question = sentence.replace(prev_word + "\s" + word, question_word)
    elif next_word != '' and search_clue_word_list(next_word, named_entity_type) != 'None':
        next_word_inf_suffix = return_inflection_suffix(next_word)
        question_word = search_rules(named_entity_type, next_word_inf_suffix)
        question = sentence.replace(word + "\s" + next_word, question_word)
    else:
        question_word = search_rules(named_entity_type, inflection_suffix)
        question = sentence.replace(word, question_word)

    if question_word != '':
        write_qa_file(writefile, question, sentence)


def search_rules(named_entity_type, inflection_suffix):
    try:
        return grammaticalrules.rules[named_entity_type][inflection_suffix]
    except KeyError:
        return ''


def search_clue_word_list(word, key):
    if word != '':
        words = word.split()
        stemmer = snowballstemmer.stemmer('tamil')
        stem_word = stemmer.stemWords(words)
        print(key)
        if words[len(words)-1] in gazetteers.clue_words.get(key):
            return word
        if stem_word[0] in gazetteers.clue_words.get(key):
            return stem_word
    return 'None'


def question_generation_ner(sentences, writefile):
    named_entities_predicted = predict_ner_tag(sentences)
    named_entity_tag_types = ['COU', 'CITY', 'CONT', 'PER', 'KIN', 'EVE', 'TRO']
    i = 0
    for sentence in sentences:
        predicted_named_entity_tags = named_entities_predicted[i]
        print(sentence)
        print(named_entities_predicted[i])
        for named_entity_tag in named_entity_tag_types:
            process_question_word(sentence, predicted_named_entity_tags, named_entity_tag, writefile)
        i = i + 1


def process_question_word(sentence, predicted_named_entity_tags, named_entity_tag, writefile):
    b_tag = "B-" + named_entity_tag
    i_tag = "I-" + named_entity_tag

    start_index = return_index_of_list(predicted_named_entity_tags, b_tag)
    end_index = start_index

    if start_index >= 0:
        while 0 <= end_index < len(predicted_named_entity_tags) - 1:
            end_index = end_index + 1
            if predicted_named_entity_tags[end_index] != i_tag:
                break
        words = sentence.split()
        named_entity = ''.join(words[start_index: end_index])
        prev_word = ''.join(words[start_index - 1])
        next_word = ''.join(words[end_index])

        question_generation(sentence, named_entity, prev_word, next_word, named_entity_tag, writefile)


def return_index_of_list(list, item):
    try:
        return list.index(item)
    except ValueError:
        return -1


def ner_question_generation(sentences, writefile):
    named_entities_predicted = predict_ner_tag(sentences)
    i = 0
    for sentence in sentences:
        named_entity_identified = False
        start_index = 0
        end_index = 0
        for index, named_entity in enumerate(named_entities_predicted[i]):
            if named_entity_identified and named_entity.startswith('B-'):
                prev_entity = extract_entities(named_entities_predicted[i][index - 1])
                identify_replacing_words(sentence, start_index, end_index, prev_entity, writefile)
                start_index = index
                end_index = index

            elif named_entity.startswith('B-'):
                start_index = index
                end_index = index
                named_entity_identified = True

            elif named_entity.startswith('I-'):
                end_index = index

            elif named_entity_identified and named_entity == 'O':
                prev_entity = extract_entities(named_entities_predicted[i][index - 1])
                identify_replacing_words(sentence, start_index, end_index, prev_entity, writefile)
                named_entity_identified = False
        i = i + 1


def extract_entities(word):
    return word[2:]


def identify_replacing_words(sentence, start_index, end_index, entity, writefile):
    words = sentence.split()
    word = ''.join(words[start_index: end_index + 1])
    prev_word = ''
    next_word = ''
    if start_index > 0:
        prev_word = ''.join(words[start_index - 1])

    if end_index < len(words) - 1:
        next_word = ''.join(words[end_index + 1])

    if entity == 'NUM' and not word.isdigit():
        process_numeric_word_question(sentence, word, prev_word, writefile)

    elif word.endswith(',') and next_word != '':
        question = gap_fill_question(sentence, words, start_index, prev_word)
        #write_qa_file(writefile, question, sentence)

    else:
        generate_question(sentence, word, prev_word, next_word, entity, writefile)


def generate_question(sentence, word, prev_word, next_word, entity, writefile):
    clue_word_entities = ['LAW', 'GOV', 'ORG', 'TRO', 'IND', 'SEC', 'SKILL', 'SOU', 'TAX', 'LOC']
    prev_clue_word = search_clue_word_list(prev_word, entity)
    inflection_suffix = get_inflection_suffix(word)

    if prev_clue_word != 'None':
        prev_word_inf_suf = return_inflection_suffix(prev_word)
        if prev_word_inf_suf == 'u"ஆகிய"' or prev_word_inf_suf == 'None':
            question_word = get_question_word(entity, prev_clue_word, inflection_suffix)
            if question_word != '':
                question = sentence.replace(prev_word + " " + word, question_word)
                write_qa_file(writefile, question, word)
                return
    clue_word = search_clue_word_list(word, entity)
    if clue_word != 'None':
        question_word = get_question_word(entity, clue_word, inflection_suffix)
        if question_word != '':
            question = sentence.replace(word, question_word)
            if question != '':
                write_qa_file(writefile, question, word)
                return

    next_clue_word = search_clue_word_list(next_word, entity)
    if next_clue_word != 'None':
        question = sentence.replace(word, 'எந்த')
        write_qa_file(writefile, question, word + " " + next_word)
        return

    if entity not in clue_word_entities:
        question_word = get_question_word(entity, clue_word, inflection_suffix)
        if question_word != '':
            question = sentence.replace(word, question_word)
            write_qa_file(writefile, question, word)
            return


def get_inflection_suffix(word):
    word_list = word.split()
    last_word = word_list[len(word_list)-1]
    return return_inflection_suffix(last_word)


def get_question_word(named_entity_type, clue_word, inflection_suffix):
    try:
        return grammaticalrules.rules[named_entity_type][inflection_suffix]
    except KeyError:
        return ''


def gap_fill_question(sentence, words, start_index, prev_word):
    part_word = check_part_word(prev_word)
    index = start_index
    question = sentence.replace(prev_word, '__________') if part_word else sentence
    no_of_words = 0
    while index < len(words):
        current_word = words[index]
        if current_word in gazetteers.multiple_named_entities:
            return question

        elif current_word.endswith(','):
            no_of_words = no_of_words + 1
            if no_of_words <= 2:
                question = question.replace(current_word, '__________,')
                no_of_words = 0
            else:
                return False

        else:
            no_of_words = no_of_words + 1
            question = question.replace(current_word, '__________')

        index = index + 1
    return False


def check_part_word(word):
    letters = tamil.utf8.get_letters(word)
    if len(letters) > 1:
        last_letter = letters[-1]
        if last_letter in mei_letters:
            return True
        tuple = tamilutils.getUnicodePoints(last_letter)
        if tuple[1] == u"அ":
            return True

        return False


def process_gap_fill_question(sentence, start_index, end_index, writefile):
    conjunction_words = ['ஆகிய', 'போன்ற', 'ஆகியன', 'என்பன', 'போன்றன', 'போன்றவை', 'ஆகியவை',
                         'என்பவை', 'மற்றும்', 'என்ற', 'என', 'முதலியன', 'முதலிய']
    words = sentence[start_index: end_index].split()
    first_replacement = False
    question = sentence
    answer = ""
    for index, word in enumerate(words):
        if not first_replacement and word.endswith(','):
            prev_word = '' if (index == 0) else words[index-1]
            if check_part_word(prev_word):
                question = question.replace(prev_word, "__________")
                answer = answer + prev_word + " "
            question = question.replace(word, "__________,")
            answer = answer + word
            first_replacement = True
        elif word.endswith(','):
            question = question.replace(word, "__________,")
            answer = answer + word
        elif first_replacement and word not in conjunction_words:
            question = question.replace(word, "__________")
            answer = answer + word + " "

    write_gap_fill_question(writefile, question, answer)


def process_numeric_word_question(sentence, word, prev_word, writefile):
    prev_word_list = ['ஆகிய', 'போன்ற', 'என்ற', 'என', 'எனப்படும்', 'பல', 'சில']

    for index, part_word in enumerate(word.split()):
        if index == len(word)-1:
            if prev_word not in prev_word_list:
                if check_if_word_is_number(word):
                    question = sentence.replace(word, 'எத்தனை')
                    write_qa_file(writefile, question, word)
                else:
                    num_inf_suffix = numeric_inflection_suffix(word)
                    if num_inf_suffix != 'None':
                        question_word = get_question_word('NUM', 'None', num_inf_suffix)
                        question = sentence.replace(word, question_word)
                        write_qa_file(writefile, question, word)
                        return

                    inf_suffix = return_inflection_suffix(word)
                    if inf_suffix != 'None':
                        question_word = get_question_word('NUM', 'None', inf_suffix)
                        question = sentence.replace(word, question_word)
                        write_qa_file(writefile, question, word)
                        return

        elif not check_if_word_is_number(word):
            return


def numeric_inflection_suffix(word):
    letters = tamil.utf8.get_letters(word)
    if len(letters) > 1:
        last_letter = letters[-1]
        second_last_letter = letters[-2]
        third_last_letter = letters[-3] if len(letters) > 2 else ''

        if last_letter == u"க":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"ஆ":
                return u"ஆக"
        elif last_letter == u"ம்":
            tuple = tamilutils.getUnicodePoints(second_last_letter)
            if tuple[1] == u"ஆ":
                return u"ஆம்"

        elif word.endswith(u"வது"):
            if third_last_letter != '':
                tuple = tamilutils.getUnicodePoints(third_last_letter)
                if tuple[1] == u"ஆ":
                    return u"ஆவது"

        return 'None'
    return 'None'
