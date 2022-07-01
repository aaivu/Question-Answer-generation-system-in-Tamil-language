import re

import gazetteers
from questiongenerator import generate_question, process_gap_fill_question
from run import write_qa_file


# regex match for எந்த, எவை question generation
def regex_match_multiple_items(sentence, writefile):
    multiple_items_regex1 = re.compile('([^,\s]+,\s)+[^,]+(?:ஆகிய\s|போன்ற\s|ஆகியன\s|என்பன\s|போன்றன\s|போன்றவை\s|ஆகியவை\s|என்பவை\s|முதலியன\s|முதலிய\s)')
    match = re.search(multiple_items_regex1, sentence)
    if match:
        process_gap_fill_question(sentence, match.start(), match.end(), writefile)
        return



# regex match for எப்போது type of question generation
def regex_match_date_time(sentence, writefile):
    date_time_regex = re.compile('(வருடம்\s)*[1-2][0-9]{3}\s*(?:இல்\s|ல்\s)')
    match = re.search(date_time_regex, sentence)
    if match:
        question = re.sub(date_time_regex, 'எந்த வருடம் ', sentence, 1)
        write_qa_file(writefile, question, match.group())

    date_time_regex2 = re.compile('[1-2][0-9]{3}(\s)*(ஆம்\s(?:வருடம்|ஆண்டு)\s){0,1}[^\s]+\s(மாதம்\s){0,1}[0-3][0-9](\s)*(?:ஆம்|ஆந்|இல்|ல்)(\s(?:திகதி|நாள்|தேதி)){0,1}')
    match2 = re.search(date_time_regex2, sentence)
    if match2:
        question = re.sub(date_time_regex2, 'எந்த வருடம், எந்த மாதம், எத்தனையாம் திகதி', sentence, 1)
        write_qa_file(writefile, question, match2.group())

    date_time_regex3 = re.compile('[1-2][0-9]{3}(\s)*-(\s)*[1-2][0-9]{3}')
    match3 = re.search(date_time_regex3, sentence)
    if match3:
        question = re.sub(date_time_regex3, 'எந்த ஆண்டு தொடக்கம் எந்த ஆண்டு', sentence, 1)
        write_qa_file(writefile, question, match3.group())

    date_time_regex3 = re.compile('[1-2][0-9]{3}(\s)*(ஆம்\s(?:வருடம்|ஆண்டு|ஆண்டில்|வருடத்தில்)\s){1}')
    match3 = re.search(date_time_regex3, sentence)
    if match3:
        question = re.sub(date_time_regex3, 'எத்தனையாம் ஆண்டு', sentence, 1)
        write_qa_file(writefile, question, match3.group())

    return False


# regex match for எத்தனை type of question generation
def regex_match_quantity(sentence, writefile):

    quantity_regex = re.compile('\d+\s*(?:ஆம்|ம்)')
    match = re.findall(quantity_regex, sentence)
    generate_numeric_question(match, sentence, 'எத்தனையாம்', writefile, True)

    quantity_regex2 = re.compile('\d+\s*(?:ஆவது|வது)')
    match2 = re.findall(quantity_regex2, sentence)
    generate_numeric_question(match2, sentence, 'எத்தனையாவது', writefile, False)

    quantity_regex3 = re.compile('\d+\s*(?:ஆக)')
    match3 = re.findall(quantity_regex3, sentence)
    generate_numeric_question(match3, sentence, 'எத்தனையாக', writefile, False)

    quantity_regex4 = re.compile('\d+\s*(?:க்கு|ற்கு|இற்கு)')
    match4 = re.findall(quantity_regex4, sentence)
    generate_numeric_question(match4, sentence, 'எத்தனைக்கு', writefile, False)

    quantity_regex5 = re.compile('\d+\s*ஆல்')
    match5 = re.findall(quantity_regex5, sentence)
    generate_numeric_question(match5, sentence, 'எத்தனையால்', writefile, False)

    quantity_regex6 = re.compile('\d+\s*(?:இல்|ல்)')
    match6 = re.findall(quantity_regex6, sentence)
    generate_numeric_question(match6, sentence, 'எத்தனையில்', writefile, True)

    quantity_regex7 = re.compile('\d+\s*ஐ')
    match7 = re.findall(quantity_regex7, sentence)
    generate_numeric_question(match7, sentence, 'எத்தனையை', writefile, False)

    quantity_regex8 = re.compile('\d+\s*உம்')
    match8 = re.findall(quantity_regex8, sentence)
    generate_numeric_question(match8, sentence, 'எத்தனையும்', writefile, False)

    quantity_regex9 = re.compile('\d+\s*இலும்')
    match9 = re.findall(quantity_regex9, sentence)
    generate_numeric_question(match9, sentence, 'எத்தனையிலும்', writefile, True)

    quantity_regex10 = re.compile('\d+\s*(?:லிருந்து|இலிருந்து)')
    match10 = re.findall(quantity_regex10, sentence)
    generate_numeric_question(match10, sentence, 'எத்தனையிலுருந்து', writefile, True)

    quantity_regex11 = re.compile('\d+\s*%')
    match11 = re.findall(quantity_regex11, sentence)
    generate_numeric_question(match11, sentence, 'எத்தனை %', writefile, False)

    quantity_regex12 = re.compile('\d+\s')
    match12 = re.findall(quantity_regex12, sentence)
    generate_numeric_question(match12, sentence, 'எத்தனை', writefile, True)

    return False


def check_gazetteer(sentence, writefile):
    words = sentence.split()
    last_index = 0
    for word_index in range(len(words)):
        if last_index == word_index:
             continue
        first_word = words[word_index]
        named_entity_type, gazetteer_val = gazetteer_search(first_word)
        if named_entity_type != 'None':
            search_val_parts = gazetteer_val.split()
            last_index = return_gazetteer_word_last_index(words, word_index, search_val_parts)
            word = ' '.join(words[word_index: last_index + 1])
            prev_word = ''.join(words[word_index - 1]) if word_index != 0 else ''
            next_word = ''.join(words[last_index + 1]) if word_index != len(words) - 1 else ''
            generate_question(sentence, word, prev_word, next_word, named_entity_type, writefile)


def return_list_index(list, item):
    try:
        return list.index(item)
    except ValueError:
        return -1


def gazetteer_search(word):
    for key, value in gazetteers.gazetteer_list.items():
        for val in value:
            part_names = val.split(" ")
            if word == part_names[0]:
                return key, val
    return 'None', 'None'


def return_gazetteer_word_last_index(wordlist, index, search_val_parts):
    i = 1
    if len(search_val_parts) > 1:
        while i < len(search_val_parts):
            if not wordlist[index + i] == search_val_parts[i]:
                break
            i = i + 1
    return index + (i - 1)


def process_single_quote(sentence, writefile):
    single_quote_regex = re.compile(r"'.*'")
    match = re.search(single_quote_regex, sentence)
    if match:
        question = re.sub(single_quote_regex, '________________', sentence)
        return write_qa_file(writefile, question, match.string)


def generate_numeric_question(match_objects, sentence, question_word, writefile, check_year_regex):
    for match_object in match_objects:
        year_regex = re.compile('[1-2][0-9]{3}')
        year_regex_match = re.search(year_regex, match_object)
        if check_year_regex and not year_regex_match:
            question = re.sub(match_object, question_word, sentence)
            write_qa_file(writefile, question, match_object)

        elif not check_year_regex:
            question = re.sub(match_object, question_word, sentence)
            write_qa_file(writefile, question, match_object)

