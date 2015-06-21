#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
__author__ = 'tysh'

"""
Открывает html-файл из корпуса Журнального зала, вынимает оттуда цитатные предложения, размечает в них цитатные границы.
В нецитатных предложениях размечает границы случайным образом для негативной выборки.
Записывает всё в один файл - обучающий корпус.
"""


import codecs, os, sys, re, csv, pickle, types, time, random, txt_util, nltk.data, nltk
from pymystem3 import Mystem
tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')  # russian splitter

from txt_util import *
mystem_object = Mystem(mystem_bin='/Users/tysh/bin/mystem')
mystem_object.start()
# global constants

quoteTree34 = pickle.load(
    open('Tree/quoteTree_34.dict', 'r'))  # quoteTree = { 'кто':{'не':{'работать':{'тот':{'__id__':1} } }}}
quoteTree = pickle.load(open('./Tree/quoteTree_5.dict', 'r'))
minquotelen = 4
idiomTree = pickle.load(open('./Tree/parenthTree.dict', 'r'))
mincollocationlen = 3


# verb_collocations = {}
# noun_collocations = {}
# verbs_stat = codecs.open('verb_collocationsnonfict.txt','w','utf-8')
# nouns_stat = codecs.open('noun_collocationsnonfict.txt','w','utf-8')


#out_sentences = codecs.open(u'training_colta.csv','w','utf-8')
#out_sentences = sys.stdout


corpus_freq = {}
document_freq = {}
document_words = set()


def skipNonWords(limit):
    """Продвигает позицию текущего слова wordsPos до конца массива слов words при отсутствии грамматического анализа в выдаче майстема (не [а-я])."""
    global words, wordsPos

    while wordsPos < limit and (words[wordsPos].get('analysis') == None or len(words[wordsPos][
                                                                                   'analysis']) == 0):  # оберег от преодоления границы, пропуск знаков препинаний и английских слов
        wordsPos += 1


# print words[0]['analysis'][0]['lex']
# {u'lex': u'\u0445\u0430\u0441', u'qual': u'bastard', u'gr': u'S,\u043c\u0443\u0436,\u043d\u0435\u043e\u0434=(\u0432\u0438\u043d,\u0435\u0434|\u0438\u043c,\u0435\u0434)'}

def findNode(Tree):
    """ Возвращает длину найденной последовательности в цитатном дереве. ищет в words, оперирует wordsPos"""
    global words, wordsPos
    wp = wordsPos
    skipNonWords(len(words))
    if wordsPos == len(words):
        wordsPos = wp  # ничего не нашли, уперлись в конец, вернули wordsPos назад как было
        return 0
    for analysis in words[wordsPos]['analysis']:
        if analysis.get('lex') != None:
            node = Tree.get(analysis['lex'])
        if node != None:
            wordsPos += 1
            return findNode(node) + 1
    wordsPos = wp # вернули назад чтобы не пропускать пунктуацию не по делу - иначе borderStretcher убегает вперед
    return 0


italics = [set(u'└'), '', set(u'┘')]
paragrs = [set(u'≈'), '', set(u'≉')]
quotemarks = [set(u'"«“\''), '', set(u'\'"»”„')]
# otherpuncts = set(u';/–…\\')
otherpuncts = set(u'[]()<>@')
# brackets = set(u'()<>')

def borderLimiter(beginning,ending):
    global words, wordsPos, Limit
    left_idiom = 0
    Limit = beginning  # for findNode2
    i = 0
    while i < beginning:
        wordsPos = i
        skipNonWords(beginning)
        k = findNode2(idiomTree)  # найти вводные слова слева и справа через findNode2 на дереве вводных слов
        if k <= 3 and left_idiom < wordsPos:
            left_idiom = wordsPos
        i += 1  # next while
    if left_idiom != 0:

        inside_idiom = ending
        Limit = ending
        wordsPos = beginning
        skipNonWords(beginning)  # beginning?
        k = findNode2(idiomTree)  # ищет вводные слова по дереву
        if k <= 3:
            right_idiom = wordsPos - k
    right_idiom = len(words)
    Limit = len(words)
    wordsPos = ending + 1
    skipNonWords(ending)
    k = findNode2(idiomTree)  # ищет вводные слова по дереву
    if k <= 3:
        right_idiom = wordsPos - k
    return right_idiom,left_idiom

    # вычислить limit для borderStretcher left & right  с учётом вводных слов


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


def findNode2(Tree):
    """ Возвращает длину найденной последовательности в дереве (например, вводных слов). ищет в words, оперирует wordsPos"""
    global words, wordsPos, Limit
    wp = wordsPos
    skipNonWords(Limit)
    if wordsPos == Limit:
        wordsPos = wp  # ничего не нашли, уперлись в конец, вернули wordsPos назад как было
        return 100
    node = Tree.get(words[wordsPos]['text'])
    if node != None:
        wordsPos += 1
        return findNode2(node) + 1
    elif Tree.get('__id__') != None:
        return 0  # нашли полную фразу

    return 200  # нашли неполную фразу или ничего вообще




def frequences(sentence):
    global document_words
    words = mystem_object.analyze(sentence)[:-1]  # запускает mystem-анализ
    for w in words:
        if w.get('analysis'):
            analysis = w['analysis'][0]
            if analysis.get('lex') :
                if corpus_freq.get(analysis['lex']) :
                    corpus_freq[analysis['lex']] += 1
                else:
                    corpus_freq[analysis['lex']] = 1
                document_words.add(analysis['lex'])


def processSentence(sentence,name):
    global wordsPos, words, Limit, balance
    id = str(hash(sentence))[:8].replace('-','') + '#' + name.split('/')[-1]
    words = mystem_object.analyze(sentence)[:-1]  # запускает mystem-анализ
    range_words = range(0, len(words))
    best_beginning = len(words)
    best_ending = 0
    quoteFound = False
    i = 0
    while i < len(words):
        wordsPos = i
        skipNonWords(len(words))
        # место для вычисления frequence freq['word[wordpos]'] = +1
        found_in_fivegrams = findNode(quoteTree) #   ищет цитаты по дереву
        if found_in_fivegrams > minquotelen:
            quoteFound = True
            if i < best_beginning :   # склейка
              best_beginning = i
            if wordsPos > best_ending :
              best_ending = wordsPos
        else:
            found_in_collocations = findNode(quoteTree34)
            if found_in_collocations >= mincollocationlen:
                quoteFound = True
                if i < best_beginning :   # склейка
                  best_beginning = i
                if wordsPos > best_ending :
                  best_ending = wordsPos

        i+=1  # next while len(words)
    if quoteFound:
        balance += 1
        right_idiom,left_idiom = borderLimiter(best_beginning,best_ending)
        beginning = borderStretcher(best_beginning,
                        left_idiom - best_beginning - 2)  # двигает границу влево до знаков препинания или разметки
        ending = borderStretcher(best_ending, right_idiom - best_ending)  # двигает границу вправо до знаков препинания или разметки
        #return beginning,ending,'1'
        write_to_corpora(beginning, ending, '1',words, out_sentences)
    else:
        if balance > 0:
            balance -= 1
            # для создания негативной выборки на каждое предложение, содержащее цитату, берется предложение, не содержащее цитаты, и ставится в случайном месте там граница с зазором в 4 слова с конца для начала квази-"цитаты" и с ограничением длины "цитаты" не меньше 4-5-6 слов
            try:
                fake_beginning = random.choice(range_words[:-8])
                fake_ending = random.choice(range_words[fake_beginning + random.choice([8, 10, 12]):])
                #return fake_beginning,fake_ending,'0'
                write_to_corpora(fake_beginning, fake_ending, '0',id, words, out_sentences)
            except:
                balance += 1

def processFile(name):
    global balance,document_freq,document_words
    start = time.time()
    try:
        document_words = set()
        f = codecs.open(name, 'r', 'utf-8').read()  # открывает исходный html-файл
        sentences = tokenizer.tokenize(txt_util.cleanHtml(f))  # убирает html теги, заменяя италик, абзацы, неразр. переносы на знаки разметки, делит на предложения
        balance = 0
        for sentence in sentences:
##            processSentence(sentence,name)
            frequences(sentence)
#            beginning, ending, answer = processSentence(sentence)
#            write_to_corpora(beginning, ending, answer, id)
        for i in list(document_words):
            if document_freq.get(i) :
                document_freq[i] += 1
            else:
                document_freq[i] = 1
    except UnicodeDecodeError:
        print name + ' UnicodeDecodeError'


def processDir(dir):
    global document_freq
    dirlist = os.listdir(dir)
    counter = len(dirlist)
    random.shuffle(dirlist)
    for filenum in range(len(dirlist)):
        if not dirlist[filenum].endswith(u'.html'):
            continue
        processFile(dir+dirlist[filenum])

        counter -= 1
        print counter
    my_freqs = codecs.open('my_freqs.csv','w','utf-8')
    N = len(dirlist)
    tfidfs = {}
#    my_freqs.write('term\ttf\tf weight\tdf\'
    for term, tf in [(k,corpus_freq[k]) for k in sorted(corpus_freq, key = corpus_freq.__getitem__, reverse = True)]:
        idf = numpy.log10(N/document_freq[term])
        tfidfs[term] = numpy.log(1+tf)*idf
        my_freqs.write('%s\t%d\t%d\t%d\t%d\n'%(term,tf,numpy.log(1+tf), document_freq[term]),log10(N/document_freq[term]))

    my_tfidfs = codecs.open('my_tfidfs.csv','w','utf-8')
    for term, tf in [(k,tfidfs[k]) for k in sorted(tfidfs, key = tfidfs.__getitem__, reverse = True)]:
        my_tfidfs.write('%s\t%d\n'%(term,tf))
    my_freqs.close()
    my_tfidfs.close()

######### begin ###########

#processFile(u'test/nlo_2006_77_ga19.html')#./downloader/test/october_2004_9_nai1.html') #'test3.html'))#
processDir(u'non_fiction/')
#processDir(u'/Users/tysh/Documents/hse/IV/citdiploma/downloader/colta/')
# for verb,positions in verb_collocations.iteritems():
#	for p in range(len(positions)):
#		verbs_stat.write(verb+';'+str(positions[p])+';'+str(p)+'\n')
# verbs_stat.close()
#
# for noun,positions in noun_collocations.iteritems():
#	for p in range(len(positions)):
#		nouns_stat.write(noun+';'+str(positions[p])+';'+str(p)+'\n')
# nouns_stat.close()


#out_sentences.close()
