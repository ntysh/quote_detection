#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'tysh'
# tfidf, document - sentence before beginning

import numpy
import codecs, os, sys, re, types, time, random
from pymystem3 import Mystem

mystem_object = Mystem(mystem_bin='/Users/tysh/bin/mystem')
mystem_object.start()

freqs = sys.stdout

quotes = codecs.open(u'true_training.txt','r','utf-8').read()

corpus_freq = {}

words = mystem_object.analyze(quotes)

for w in words:
    if w.get('analysis'):
        analysis = w['analysis'][0]
        if analysis.get('lex') :
            if corpus_freq.get(analysis['lex']) :
                corpus_freq[analysis['lex']] += 1
            else:
                corpus_freq[analysis['lex']] = 1

corpora = quotes.split('\n')
N = len(corpora)
document_freq = {}

for k,line in enumerate(corpora):
    answer = int(line[0])
    if answer != 1 : continue
    words = mystem_object.analyze(line[1:])  # запускает mystem-анализ
    beginning,ending = [words.index(word) for word in words if '|' in word['text']]
    document_words = set()
    for w in words[beginning-4:beginning]:
        if w.get('analysis'):
            analysis = w['analysis'][0]
            if analysis.get('lex') :
                document_words.add(analysis['lex'])

    for i in list(document_words):
        if document_freq.get(i) :
            document_freq[i] += 1
        else:
            document_freq[i] = 1

#for term, tf in [(k,corpus_freq[k]) for k in sorted(corpus_freq, key = corpus_freq.__getitem__, reverse = True)]:
for term, df in document_freq.items():
    try:
        tf = corpus_freq[term]
        idf = numpy.log10(N/float(df))
        tfidf = numpy.log(1+tf)*idf
        freqs.write('%s\t%d\t%d\t%d\n'%(term,tf,tfidf, df))
    except:
  #      print "problem term", term
        pass
