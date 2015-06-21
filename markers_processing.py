#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'


import codecs, os, sys, re, types, time, random
import cPickle as pickle
from pymystem3 import Mystem

FINDER_LIMIT = 14 #изучать границы не уходя дальше 7 слов от цитатных границ
EDGE_BUFFER = 8
POSITIVE = 1
NO_QUOTE = 0
BAD_BORDER = 2
NEAR = 3

mystem_object = Mystem(mystem_bin='/Users/tysh/bin/mystem')
mystem_object.start()

def useful_lists(name,Cap=False):
    out = []
    for i in [line.rstrip('\n') for line in codecs.open("./useful_lists/"+name+'.txt', 'r', 'utf-8').readlines()]:
        if Cap :
            out.append(i.lower().capitalize())
        else:
            out.append(i.replace('\s', ''))
    return set(out)

mention_verbs = useful_lists('top_nonf_v_syns')
mention_nouns = useful_lists('top_nonf_n')
authors       = useful_lists('authors')  #.
key_fictions = useful_lists('key_fiction')
key_nonfictions = useful_lists('key_othernonfiction')
freqs = useful_lists('top_freqs')
bibleisms = useful_lists('bibleisms', Cap = True)
tfidfs_near = useful_lists('top_tfidf_near')
dfs_near = useful_lists('top_df_near')


part_of_speech_tags = {'A', 'ADV', 'ANUM', 'CONJ', 'NUM', 'PART', 'PR', 'S', 'SPRO', 'V', 'ADVPRO', 'INTJ',
                       'APRO'}  # 'COM',
mystem_other = {'parenth', 'geo', 'abbr', 'famn', 'bastard', 'inpraes', 'praet', 'inf', 'partcp', 'indic',
                'imper', 'act', 'pass', 'anim', 'inan', 'tran', 'intr', 'latin', 'persn', 'patrn'}
#                'inform', 'ger', 'poss', 'comp', 'persn', 'obsol', 'number', 'supr','praes'}  # not mystem.other actually
# ,

punctuations = [set(u'"«“\'()[]@/–…:,.?!≉≈└'), '', set(u'\'"»()[]@/–…:,.?!≉≈┘')]

markers_list = part_of_speech_tags | mystem_other | punctuations[0] | punctuations[2] | {'author', 'key_f', 'key_nf', 'm_verb', 'bibl', 'm_noun',
                      'ss', 'vv','sv', 'vs', 'as', 'sa', 'advprov', 'vadvpro', 'aprov', 'vapro', 'sprov', 'vspro', 'spros', 'sspro', 'advpros', 'sadvpro', 'apros', 'sapro',
                                                                                         'rare','tfidfn','dfn'}

num = re.compile('[0-9]+')

def grSplit(analysisgr):
    """
    Получение массива грамматических тэгов без повтора для каждой гипотезы
    :param analysisgr: analysis['gr']
    :return:  set of gr's
    """
    return set(analysisgr.replace('(', '').replace(')', '').replace('==', '=').replace('=', ',').replace('|', ',').split(','))

def grDistance(markers, pos, limit, words):
    """
        Возвращает словарь, где ключи - маркеры, а значения - кортежи расстояний от границы цитаты pos.
        Кортеж из одного 100500, если в предложении вообще маркера нет.
        limit - граница окна
    """


    out = dict.fromkeys(markers,100500)  # k - marker, v - distances tuple
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit
    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') == None:
            word = words[i]['text'].replace(' ', '').replace('\\t', '')
            if len(word) == 0:
                continue
            if num.search(word) != None:  # поиск цифр
                if out.get('number') != None:
                    out['number'] = p

            for k in list(set(word) & punctuations[direction + 1]):  # поиск пунктуации
                out[k] = p

        else:
            #	print words[i]['text']
            for analysis in words[i]['analysis']:
                if analysis.get('gr') != None:  # поиск грамматических тэгов
                    #	PoS = analysis['gr'].replace('=',',').split(',')[0]

                    #			print len(analysis['gr'])
                    for gram in list(grSplit(analysis['gr'])):
                        if out.get(gram) :
                            out[gram] = p

                    if analysis.get('qual') :  # поиск несловарного слова
                        out['bastard'] = p

                    #   будет работать, если для английских слов ['analysis'] == [], а не None
                else:
                    out['latin'] = p
    return out


def seqDistance(sequence, pos, limit, words):
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit
    seq = sequence[::direction]
    j = 0
    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') != None:
            for analysis in words[i]['analysis']:
                word_class = analysis['gr'].replace('=', ',').split(',')[0]
                coef = (p // 2 + 1) * direction + 4
                #				if word_class == 'V':
                #					if verb_collocations.get(analysis['lex']) == None :
                #						verb_collocations[analysis['lex']]=[0,0,0,0,0,0,0,0,0]
                #					verb_collocations[analysis['lex']][coef] += 1
                #				if word_class == 'S':
                #					if noun_collocations.get(analysis['lex']) == None :
                #						noun_collocations[analysis['lex']]=[0,0,0,0,0,0,0,0,0]
                #					noun_collocations[analysis['lex']][coef] += 1
                # print '==',word_class,p,j
                if seq[j] == word_class:
                    if j == len(seq) - 1:
                        return p - len(seq)
                    else:
                        j += 1
                else:
                    j = 0  # continue? надо пройти все разборы, если нужная часть речи есть хоть в одном!
    return 100500


def idiomDistance(sequence, pos, limit, words):
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit
    seq = sequence[::direction]
    j = 0
    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') != None:
            word = words[i]['text']
            if seq[j] == word:
                if j == len(seq) - 1:
                    return p - len(seq)
                else:
                    j += 1
            else:
                j = 0
    return 100500


def textDistance(array, pos, limit, words):
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit
    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') != None:
            word = words[i]['text'].replace(' ', '').replace('\\t', '')
            if word in array:
                return p
    return 100500


def lexDistance(array, pos, limit, words):
    if limit <= 0:
        direction = -1
        r = -limit
    else:
        direction = 1
        r = limit
    for p in range(0, r):
        i = pos + p * direction
        if words[i].get('analysis') != None:
            for analysis in words[i]['analysis']:
                lex = analysis['lex']
                if lex in array:
                    return p
    return 100500

def findMarkers(words, border,limit):

    out = grDistance(markers_list,border,limit, words)  # задает список маркеров и окно функции сбора расстояний: начало - отредактированное начало цитаты, конец -	начало предложения, если оно не дальше 12 токенов влево (примерно пяти слов)
    out['author'] = lexDistance(authors, border, limit, words)
    out['bibleism'] = textDistance(bibleisms, border, limit, words)
    out['key_fiction']=lexDistance(key_fiction, border, limit, words)
    out['key_nonfiction']=lexDistance(key_other_nonfiction, border, limit, words) # key_fictionl
    out['mention_verb']=lexDistance(mention_verbs, border, limit, words)
    out['mention_noun']=lexDistance(mention_nouns, border, limit, words)
    """
    out['sv']=seqDistance(['S', 'V'], border, limit, words)
    out['vs']=seqDistance(['V', 'S'], border, limit, words)
    out['ss']=seqDistance(['S', 'S'], border, limit, words)
    out['vv']=seqDistance(['V', 'V'], border, limit, words)
    out['as']=seqDistance(['A', 'S'], border, limit, words)
    out['sa']=seqDistance(['S', 'A'], border, limit, words)
    out['av']=seqDistance(['A', 'V'], border, limit, words)
    out['va']=seqDistance(['V', 'A'], border, limit, words)
    out['advprov']=seqDistance(['ADVPRO', 'V'], border, limit, words)
    out['vadvpro']=seqDistance(['V', 'ADVPRO'], border, limit, words)
    out['aprov']=seqDistance(['APRO', 'V'], border, limit, words)
    out['vapro']=seqDistance(['V', 'APRO'], border, limit, words)
    out['sprov']=seqDistance(['SPRO', 'V'], border, limit, words)
    out['vspro']=seqDistance(['V', 'SPRO'], border, limit, words)
    out['spros']=seqDistance(['SPRO', 'S'], border, limit, words)
    out['sspro']=seqDistance(['S', 'SPRO'], border, limit, words)
    out['advpros']=seqDistance(['ADVPRO', 'S'], border, limit, words)
    out['sadvpro']=seqDistance(['S', 'ADVPRO'], border, limit, words)
    out['apros']=seqDistance(['APRO', 'S'], border, limit, words)
    out['sapro']=seqDistance(['S', 'APRO'], border, limit, words)
    """
    return out

chains = {'S': ('V','S','A','SPRO','APRO', 'ADVPRO'), 'V' : ('S','V', 'A', 'SPRO','APRO', 'ADVPRO'),
          'APRO': ('V','S'), 'ADVPRO': ('V','S'), 'SPRO': ('V','S'), 'A': ('V','S')}

def locateMarkers(words, border,limit):
    out = [[] for i in range(len(words))]
    nextseq = []
    for i,w in enumerate(words):
        if w.get('analysis') :
            if w['analysis'] == [] : #   будет работать, если для английских слов ['analysis'] == [], а не None
               out[i].append('latin')
            for analysis in w['analysis']:
                if analysis.get('gr') :
                    out[i] += list(grSplit(analysis['gr']) & markers_list)
                    word_class = analysis['gr'].replace('=', ',').split(',')[0]
                    if nextseq :
                        if word_class in nextseq:
                            out[prevseq].append((prev+word_class).lower())
                            nextseq = []
                    elif chains.get(word_class):
                        nextseq = chains[word_class]
                        prevseq = i
                        prev = word_class
                if analysis.get('qual') :  # поиск несловарного слова
                    out[i].append('bastard')
                if analysis['lex'] in authors:  out[i].append('author')
                if analysis['lex'] in key_fictions:  out[i].append('key_f')
                if analysis['lex'] in key_nonfictions:  out[i].append('key_nf')
                if analysis['lex'] in bibleisms:  out[i].append('bibl')
                if analysis['lex'] in mention_verbs:  out[i].append('m_verb')
                if analysis['lex'] in mention_nouns:  out[i].append('m_noun')
                if analysis['lex'] in tfidfs_near:  out[i].append('tfidfn')
                if analysis['lex'] in dfs_near:  out[i].append('dfn')
                if analysis['lex'] not in freqs:  out[i].append('rare')
        else:
            for k in list(set(w['text']) & markers_list):  # поиск пунктуации
                out[i].append(k)

    return out


def binFragments1(b,e,situ):
    fragments = [0,max(b-NEAR,0),b,e,min(e+NEAR,len(situ)),len(situ)]
    vect = []
    for ff in range(len(fragments)-1):
        dic = dict.fromkeys(markers_list,0)
        for marker in markers_list:
            for s in situ[fragments[ff]:fragments[ff+1]]:
                if marker in s :
                    dic[marker] = 1
        vect += [value for (key, value) in sorted(dic.items())]
    return vect

def distanceFragments(b,e,situ):
    fragments = [(0,b),(b,e),(e,len(situ))]
    be = [b,0,e]
    vect = []
    for ff in [-1,0,1]:
        dic = dict.fromkeys(markers_list,0)
        for marker in markers_list:
            for i,s in enumerate(situ[fragments[ff+1][0]:fragments[ff+1][1]]):
                if marker in s :
                    dic[marker] = 1 + fragments[ff+1][0] + (i  - be[ff+1])*ff  #  b-i or 0 or i-e
        vect += [value for (key, value) in sorted(dic.items())]

    return vect


f1 = markers_list - {'tfidfn','dfn'}
f2 = markers_list
f3 = markers_list - {'tfidfn','dfn'}
f4 = markers_list - {'tfidfn','dfn'}
f5 = markers_list - {'tfidfn','dfn'}

def binFragments(b,e,situ):
    vect = []
    lsitu = len(situ)
    ll = max(b-NEAR,0)
    rr = min(e+NEAR,lsitu)
    for l,r,ml in [(0,ll,f1),(ll,b,f2),(b,e,f3),(e,rr,f4),(rr,lsitu,f5)]: # sync with featureNames!
        dic = dict.fromkeys(ml,0)
        for marker in ml:
            for i,s in enumerate(situ[l:r]):
                if marker in s :
                    dic[marker] = 1
        vect += [value for (key, value) in sorted(dic.items())]

    return vect

def featuresNames():
    vect = []
    ll,rr,b,e,lsitu = 0,0,0,0,0
    i = 1
    for l,r,ml in [(0,ll,f1),(ll,b,f2),(b,e,f3),(e,rr,f4),(rr,lsitu,f5)]: # sync with binFragments!
        dic = dict.fromkeys(ml,0)
        vect += [key+str(i) for (key, value) in sorted(dic.items())]
        i += 1
    return vect
