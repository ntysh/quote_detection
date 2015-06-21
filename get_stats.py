#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'

from markers_processing import *
import numpy as np
from sklearn import svm
from random import randint

FINDER_LIMIT = 14 #изучать границы не уходя дальше 7 слов от цитатных границ
EDGE_BUFFER = 8
POSITIVE = 1
NO_QUOTE = 0
BAD_BORDER = 2
NEGATIVES = 1

shorts = 0

def markers(words,pos,limit,inside):
    mrk = findMarkers(words,pos,limit)
    mrk.update(inside)
    return [value for (key, value) in sorted(mrk.items())]

def makeMatrix(corpora):
    global shorts
    for k,line in enumerate(corpora[1:]):
        answer, beginning, ending, ids, name, sentence = line.rstrip('\n').split('#')
        answer, beginning, ending = int(answer), int(beginning), int(ending)

        words = mystem_object.analyze(sentence.replace('|',''))[:-1]  # запускает mystem-анализ

        #if answer == NO_QUOTE and k%2 : continue # негативных N/2

        inside_markers = insideMarkers(words,beginning, ending - beginning)
        left_matrix.append(markers(words,beginning - 1,max(0 - beginning, -FINDER_LIMIT),inside_markers))
        right_matrix.append(markers(words,ending - 1,min(len(words) - ending, FINDER_LIMIT),inside_markers))
        target.append(answer)

        if answer == POSITIVE:
            for i in range(NEGATIVES): #if k % 2 : # негативных N/2
                try:
                    b = randint(0,len(words)-EDGE_BUFFER)
                except ValueError: #короткие предложения не учитываются в негативной выборке неправильных границ
                    shorts +=1
                    break
                e = randint(b+1,len(words))
                inside_markers = insideMarkers(words,b, e-b)
                left_matrix.append(markers(words,b - 1,max(0 - b, -FINDER_LIMIT),inside_markers))
                right_matrix.append(markers(words,e - 1,min(len(words) - e, FINDER_LIMIT),inside_markers))
                target.append(BAD_BORDER)

    X_left = np.array(left_matrix)
    X_right = np.array(right_matrix)
    y = np.array(target)
    return X_left,X_right,y


# == begin
t1 = time.time()

clf_left = svm.SVC(probability=True,class_weight='auto',kernel = 'rbf')
clf_right = svm.SVC(probability=True,class_weight='auto',kernel = 'rbf')

left_matrix = []
right_matrix = []
target = []

quotes_in_context = codecs.open(u'corpora_quotes_in_sentences.csv','r','utf-8').readlines() #answer, beginning, ending, ids, name, sentence                                                                        #    0#12#1#-6515558#nlo_2006_77_ga19.html|Я себя советским чувствую заводом, вырабатывающим счастье|.
testing_quotes = codecs.open(u'testing_quotes_in_sentences.csv','r','utf-8').readlines() #answer, beginning, ending, ids, name, sentence                                                                        #    0#12#1#-6515558#nlo_2006_77_ga19.html|Я себя советским чувствую заводом, вырабатывающим счастье|.



#TRAINING DATA
X_left,X_right,y = makeMatrix(quotes_in_context)

t2 = time.time()
print 'makeMatrix training',t2-t1


print 'short sentences number = ', shorts

clf_left = clf_left.fit(X_left,y)
clf_right = clf_right.fit(X_right,y)
t3 = time.time()
print 'Training',t3-t2

pickle.dump(clf_left, open('classifier_left.pickle','wb'))
pickle.dump(clf_right, open('classifier_right.pickle','wb'))

#TESTING DATA

X_left_test,X_right_test,correct = makeMatrix(testing_quotes)

t4 = time.time()
print 'makeMatrix testing',t4-t3

left_prediction = clf_left.predict(X_left_test)
right_prediction = clf_right.predict(X_right_test)

t5 = time.time()
print 'testing',t5-t4

#open('l_prediction.tsv','wb').write(list_to_str(left_prediction))
#open('r_prediction.tsv','wb').write(list_to_str(right_prediction))
#open('correct.tsv','wb').write(list_to_str(correct))

score1 = sum([int(left_prediction[i] == int(correct[i])) for i in range(len(correct))])
score2 = sum([int(right_prediction[i] == int(correct[i])) for i in range(len(correct))])
print('left classifier accuracy: {} ({}/{})'.format(float(score1)/len(correct), score1, len(correct))),\
    ('right classifier accuracy: {} ({}/{})'.format(float(score2)/len(correct), score2, len(correct)))


print time.time()-t1