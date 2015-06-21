#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'

from txt_util import *
from markers_processing import *
from sklearn.svm import SVC
from itertools import combinations
import numpy as np
import nltk
tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')  # russian splitter


out = sys.stdout

quote_machine = pickle.load(open('quote_classifier.pickle','r'))
print "quote machine loaded"
borders_machine = pickle.load(open('borders_classifier.pickle','r'))
print "borders machine loaded"

def runMachine(machine,situ,words):
     out = []
     for l,r in combinations(range(len(situ)),2):
        if r-l < EDGE_BUFFER : continue  #  надо больше ограничений
        if not words[l].get('analysis') or not words[r].get('analysis'): continue
        out.append(l,r,list(quote_machine.predict_proba(distanceFragments(l,r,situ))[0]))
     return out


maxprob = [0.97,0.6]

def processFile(name):
    text = codecs.open(name, 'r', 'utf-8').read()  # открывает исходный html-файл
    sentences = tokenizer.tokenize(cleanHtml(text)) #Возвращает предложения со знаками переноса в конце.
#    sentences = text.split('\n')
#    sentences = tokenizer.tokenize(text)
    quotes_found = 0
    for c,sentence in enumerate(sentences):
        words = mystem_object.analyze(sentence)[:-1]  # запускает mystem-анализ
        lwords = len(words)

        if lwords <= EDGE_BUFFER :
            continue

        markers_in_situ = locateMarkers(words, 0, lwords)
        quotecombinations = []

        for l,r in combinations(range(lwords),2):
            if r-l < EDGE_BUFFER : continue  #  надо больше ограничений
            if not words[l].get('analysis') or not words[r].get('analysis'): continue
            #перечисление всех машин - сейчас запускаются все
            for i,hypos in enumerate([list(quote_machine.predict_proba(binFragments(l,r,markers_in_situ))[0])
                                    ]):
                m = max(hypos)
                case = hypos.index(max(hypos))
                if case == POSITIVE and m > 0.75 :
                    quotecombinations.append((l,r))

        best_prob = 0
        bestcase = None
        for l,r in quotecombinations:
            hypos = list(borders_machine.predict_proba(binFragments(l,r,markers_in_situ))[0])
            m = max(hypos)
            case = hypos.index(max(hypos))
            if case == POSITIVE and m > 0.75 :
#                print_with_borders(out,l,r, str(1), m, words)
                best_prob = m
                bestlr = (l,r)

        if best_prob: print_with_borders(out,bestlr[0],bestlr[1], str(1), best_prob, words)
    return quotes_found

def processDir(dir):
    dirlist = os.listdir(dir)
    max_quotes_in_file = ''
    max_quote_found = 0
    for filename in dirlist:
        if filename.startswith(u'.'): continue
        current_quote_found = processFile(dir+filename)
        print current_quote_found, filename
        if current_quote_found > max_quote_found:
            max_quote_found = current_quote_found
            max_quotes_in_file = filename
            print 'new max_quote_found',max_quote_found

    print str(max_quote_found) +' quotes in file ' + max_quotes_in_file


#processFile(u'test/test_ga19.html')
#processFile(u'true_testing.txt')
#processFile(u'testing_quotes_in_sentences.csv')

#processDir(u'/Users/tysh/Documents/hse/IV/citdiploma/downloader/colta/')
processDir(u'/Users/tysh/Documents/hse/IV/citdiploma/downloader/fiction/')
#processDir(u'/Users/tysh/Documents/hse/IV/citdiploma/news_txt/')
