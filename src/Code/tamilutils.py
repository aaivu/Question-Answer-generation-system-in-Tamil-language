import tamil

def findinflectionsuffix(word):
    letters = tamil.utf8.get_letters(word)
    if len(letters) > 1:
        lastletter = letters[-1]
        secondlastletter = letters[-2]
        lastletterunicode = getUnicodePoints(lastletter)
        thirdlastletter = letters[-3] if len(letters) > 2 else ''
        forthlastletter = letters[-4] if len(letters) > 3 else ''

        if lastletter == u"கு":
            return u"கு"
        elif lastletter == u"ன்":
            tuple = getUnicodePoints(secondlastletter)
            if tuple[1] == u"இ":
                return u"இன்"
        elif lastletter == u"ல்":
            tuple = getUnicodePoints(secondlastletter)
            if tuple[1] == u"ஆ":
                return u"ஆல்"
            elif tuple[1] == u"இ":
                return u"இல்"
        elif lastletter == u"து":
            tuple = getUnicodePoints(secondlastletter)
            if tuple[1] == u"அ":
                return u"அது"
        elif lastletter == u"ம்":
            tuple = getUnicodePoints(secondlastletter)
            if tuple[1] == u"உ":
                return u"உம்"
        elif lastletter == u"ன":
            tuple = getUnicodePoints(secondlastletter)
            if tuple[1] == u"ஆ":
                return u"ஆன"
        elif lastletterunicode[1] == u"ஐ":
            return u"ஐ"
        elif word.endswith(u"டைய"):
            if thirdlastletter != '':
                tuple = getUnicodePoints(thirdlastletter)
                if tuple[1] == u"உ":
                    return u"உடைய"
        elif word.endswith(u"டன்"):
            if thirdlastletter != '':
                tuple = getUnicodePoints(thirdlastletter)
                if tuple[1] == u"உ":
                    return u"உடன்"
        elif word.endswith(u"டன்"):
            if thirdlastletter != '':
                tuple = getUnicodePoints(thirdlastletter)
                if tuple[1] == u"உ":
                    return u"உடன்"
        elif word.endswith(u"கிய"):
            if thirdlastletter != '':
                tuple = getUnicodePoints(thirdlastletter)
                if tuple[1] == u"ஆ":
                    return u"ஆகிய"
        elif word.endswith(u"ருந்து"):
            if forthlastletter != '':
                tuple = getUnicodePoints(forthlastletter)
                if tuple[1] == u"இ":
                    return u"இருந்து"
        elif word.endswith('மீது'):
            return u"மீது"

        return 'None'
    return 'None'

def getUnicodePoints(letter):
    try:
        # when uyir letters are passed it will throw an IndexError
        uyir_letters = tamil.utf8.uyir_letters
        rval = tamil.utf8.splitMeiUyir(letter)
        if not isinstance(rval, tuple):
            if rval in uyir_letters:
                print("This is an uyir letter")
                return (u'', rval)
            print(rval)
            return (rval, u'')
        print("Valid")
        print(rval)
        return rval
    except IndexError as idxerr:
        pass
    except ValueError as valerr:
        # non tamil letters cannot be split - e.g. '26வது'
        pass
        # could be english string etc. multiple-letter (word-like) input etc

    return (u'', u'')
