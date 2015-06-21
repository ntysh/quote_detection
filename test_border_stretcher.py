#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'

def borderStretcher(pos, limit):
    """Исправляет границы цитаты на ближайшие к одному из знаков препинания, среди которых есть иерархия. Возвращает позицию границы"""
    global words
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit

    qmpos1 = 0
    qmpos2 = 0
    colonpos = 0
    nbsppos = 0
    parapos = 0
    italpos = 0
    otherpos = 0
    commapos = 0

    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') == None:
            word = words[i]['text'].replace(' ', '').replace('\\t', '')
            if len(word) == 0:
                continue
            if len(set(word) & quotemarks[direction + 1]) != 0:
                if qmpos1 == 0:
                    qmpos1 = i - direction
            elif u'■' in word:
                if nbsppos == 0:
                    nbsppos = i - direction
            elif len(set(word) & paragrs[direction + 1]) != 0:
                if parapos == 0:
                    parapos = i - direction
            elif u':' in word:
                if colonpos == 0:
                    colonpos = i - direction
            elif len(set(word) & quotemarks[abs(direction - 1)]) != 0:
                if qmpos2 == 0:
                    qmpos2 = i - direction
            elif len(set(word) & italics[direction + 1]) != 0:
                if italpos == 0:
                    italpos = i - direction
            elif len(set(word) & otherpuncts) != 0:
                if otherpos == 0:
                    otherpos = i - direction
            elif u',' in word:
                if commapos == 0:
                    commapos = i - direction
    if qmpos1 != 0:
        return qmpos1
    if nbsppos != 0:
        return nbsppos
    elif parapos != 0:
        return parapos
    elif colonpos != 0:
        return colonpos
    elif qmpos2 != 0:
        return qmpos2
    elif italpos != 0:
        return italpos
    elif otherpos != 0:
        return otherpos
    elif commapos != 0:
        return commapos
    if direction == 1:
        return pos - 1
    else:
        return pos

from pymystem3 import Mystem
mystem_object = Mystem(mystem_bin='/Users/tysh/bin/mystem')
s = u'Они поняли [лагерную] мудрость “не верь, не бойся, не проси” <…> И вослед изначальным носителям этой мудрости предпочитают не работать на государство, избегать любых договоров с ним”.'
words = mystem_object.analyze(s)
limit = len(words)-17
ending = 16
italics = [set(u'└'), '', set(u'┘')]
paragrs = [set(u'≈'), '', set(u'≉')]
quotemarks = [set(u'"«“\''), '', set(u'\'"»”„')]
# otherpuncts = set(u';/–…\\')
otherpuncts = set(u'[]()<>@')
# brackets = set(u'()<>')

end = borderStretcher(ending, limit) #est 18

print words[end-1]['text'], words[end]['text'],words[end+1]['text']