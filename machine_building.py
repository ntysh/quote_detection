#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'


from markers_processing import *
import numpy as np
from sklearn import svm
from random import randint
import matplotlib.pyplot as plt
#from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from io import StringIO
import pydot
#marks = ['ADV1', 'bibleism1', 'partcp1', 'obsol1', 'vv1', 'inpraes1', 'vspro1', 'pass1', 'indic1', 'sv1', u'\u20131', 'PART1', u'\u25141', 'famn1', 'intr1', u'\u25181', 'CONJ1', u'.1', 'inf1', 'ANUM1', 'PR1', 'bastard1', u'"1', u"'1", u'\u20261', u')1', u'(1', u'\xab1', u']1', u',1', u'/1', 'parenth1', 'inan1', 'sadvpro1', 'apros1', 'NUM1', 'author1', 'mention_noun1', 'vadvpro1', u'\xbb1', u':1', u'\u201c1', u'?1', 'A1', u'@1', 'ADVPRO1', 'anim1', 'sapro1', 'vapro1', u'\u22491', u'\u22481', 'APRO1', 'persn1', 'non-cyrillic1', 'INTJ1', 'S1', u'!1', 'abbr1', 'V1', 'aprov1', u'[1', 'key_fiction1', 'geo1', 'praet1', 'sspro1', 'praes1', 'ss1', 'spros1', 'tran1', 'as1', 'imper1', 'sprov1', 'SPRO1', 'key_nonfiction1', 'mention_verb1', 'vs1', 'act1', 'advpros1', 'sa1', 'advprov1', 'patrn1', 'ADV2', 'bibleism2', 'partcp2', 'obsol2', 'vv2', 'inpraes2', 'vspro2', 'pass2', 'indic2', 'sv2', u'\u20132', 'PART2', u'\u25142', 'famn2', 'intr2', u'\u25182', 'CONJ2', u'.2', 'inf2', 'ANUM2', 'PR2', 'bastard2', u'"2', u"'2", u'\u20262', u')2', u'(2', u'\xab2', u']2', u',2', u'/2', 'parenth2', 'inan2', 'sadvpro2', 'apros2', 'NUM2', 'author2', 'mention_noun2', 'vadvpro2', u'\xbb2', u':2', u'\u201c2', u'?2', 'A2', u'@2', 'ADVPRO2', 'anim2', 'sapro2', 'vapro2', u'\u22492', u'\u22482', 'APRO2', 'persn2', 'non-cyrillic2', 'INTJ2', 'S2', u'!2', 'abbr2', 'V2', 'aprov2', u'[2', 'key_fiction2', 'geo2', 'praet2', 'sspro2', 'praes2', 'ss2', 'spros2', 'tran2', 'as2', 'imper2', 'sprov2', 'SPRO2', 'key_nonfiction2', 'mention_verb2', 'vs2', 'act2', 'advpros2', 'sa2', 'advprov2', 'patrn2', 'ADV3', 'bibleism3', 'partcp3', 'obsol3', 'vv3', 'inpraes3', 'vspro3', 'pass3', 'indic3', 'sv3', u'\u20133', 'PART3', u'\u25143', 'famn3', 'intr3', u'\u25183', 'CONJ3', u'.3', 'inf3', 'ANUM3', 'PR3', 'bastard3', u'"3', u"'3", u'\u20263', u')3', u'(3', u'\xab3', u']3', u',3', u'/3', 'parenth3', 'inan3', 'sadvpro3', 'apros3', 'NUM3', 'author3', 'mention_noun3', 'vadvpro3', u'\xbb3', u':3', u'\u201c3', u'?3', 'A3', u'@3', 'ADVPRO3', 'anim3', 'sapro3', 'vapro3', u'\u22493', u'\u22483', 'APRO3', 'persn3', 'non-cyrillic3', 'INTJ3', 'S3', u'!3', 'abbr3', 'V3', 'aprov3', u'[3', 'key_fiction3', 'geo3', 'praet3', 'sspro3', 'praes3', 'ss3', 'spros3', 'tran3', 'as3', 'imper3', 'sprov3', 'SPRO3', 'key_nonfiction3', 'mention_verb3', 'vs3', 'act3', 'advpros3', 'sa3', 'advprov3', 'patrn3', 'ADV4', 'bibleism4', 'partcp4', 'obsol4', 'vv4', 'inpraes4', 'vspro4', 'pass4', 'indic4', 'sv4', u'\u20134', 'PART4', u'\u25144', 'famn4', 'intr4', u'\u25184', 'CONJ4', u'.4', 'inf4', 'ANUM4', 'PR4', 'bastard4', u'"4', u"'4", u'\u20264', u')4', u'(4', u'\xab4', u']4', u',4', u'/4', 'parenth4', 'inan4', 'sadvpro4', 'apros4', 'NUM4', 'author4', 'mention_noun4', 'vadvpro4', u'\xbb4', u':4', u'\u201c4', u'?4', 'A4', u'@4', 'ADVPRO4', 'anim4', 'sapro4', 'vapro4', u'\u22494', u'\u22484', 'APRO4', 'persn4', 'non-cyrillic4', 'INTJ4', 'S4', u'!4', 'abbr4', 'V4', 'aprov4', u'[4', 'key_fiction4', 'geo4', 'praet4', 'sspro4', 'praes4', 'ss4', 'spros4', 'tran4', 'as4', 'imper4', 'sprov4', 'SPRO4', 'key_nonfiction4', 'mention_verb4', 'vs4', 'act4', 'advpros4', 'sa4', 'advprov4', 'patrn4', 'ADV5', 'bibleism5', 'partcp5', 'obsol5', 'vv5', 'inpraes5', 'vspro5', 'pass5', 'indic5', 'sv5', u'\u20135', 'PART5', u'\u25145', 'famn5', 'intr5', u'\u25185', 'CONJ5', u'.5', 'inf5', 'ANUM5', 'PR5', 'bastard5', u'"5', u"'5", u'\u20265', u')5', u'(5', u'\xab5', u']5', u',5', u'/5', 'parenth5', 'inan5', 'sadvpro5', 'apros5', 'NUM5', 'author5', 'mention_noun5', 'vadvpro5', u'\xbb5', u':5', u'\u201c5', u'?5', 'A5', u'@5', 'ADVPRO5', 'anim5', 'sapro5', 'vapro5', u'\u22495', u'\u22485', 'APRO5', 'persn5', 'non-cyrillic5', 'INTJ5', 'S5', u'!5', 'abbr5', 'V5', 'aprov5', u'[5', 'key_fiction5', 'geo5', 'praet5', 'sspro5', 'praes5', 'ss5', 'spros5', 'tran5', 'as5', 'imper5', 'sprov5', 'SPRO5', 'key_nonfiction5', 'mention_verb5', 'vs5', 'act5', 'advpros5', 'sa5', 'advprov5', 'patrn5']

NEGATIVES = 1
mode = 'bin' #'bin' or 'distance'
shorts = 0


def makeMatrix(corpora,mode):
    global shorts
    matrix = []
    target = []
    for k,line in enumerate(corpora):
#        answer, beginning, ending, sentence = line.rstrip('\n').split('#')
#        answer, beginning, ending = int(answer), int(beginning), int(ending)
        answer = int(line[0])
        words = mystem_object.analyze(line[1:-1])  # запускает mystem-анализ
        beginning,ending = [words.index(word) for word in words if '|' in word['text']]

        lwords = len(words)
        markers_in_situ = locateMarkers(words, 0, lwords)

        if mode == "quote" :
            if answer == BAD_BORDER  : continue
            matrix.append(binFragments(beginning,ending,markers_in_situ))
            target.append(answer)
        else:
            if answer == NO_QUOTE  : continue
            matrix.append(binFragments(beginning,ending,markers_in_situ))
            target.append(answer)
            for i in range(NEGATIVES):
                if lwords > EDGE_BUFFER: #короткие предложения не учитываются в негативной выборке неправильных границ
                    while True :
                        b = randint(0,lwords-EDGE_BUFFER)
                        if b != beginning : break
                else:
                    shorts +=1
                    break
                while True :
                    e = randint(b+1,lwords)
                    if e != ending : break
                matrix.append(binFragments(b,e,markers_in_situ))
                target.append(NO_QUOTE)


    print mode,' corpora len =',len(target)
    print mode,' features num = ', len(matrix[0])
    X = np.array(matrix)
    y = np.array(target)
    return X, y


def list_to_str(x):
    result = [str(y) for y in x]
    return '\t'.join(result)

#training_quotes = codecs.open(u'training_quotes_in_sentences.csv','r','utf-8').readlines() #answer, beginning, ending, sentence                                                                        #    0#12#1#-6515558#nlo_2006_77_ga19.html|Я себя советским чувствую заводом, вырабатывающим счастье|.
#testing_quotes = codecs.open(u'testing_quotes_in_sentences.csv','r','utf-8').readlines() #answer, beginning, ending, sentence                                                                        #    0#12#1#-6515558#nlo_2006_77_ga19.html|Я себя советским чувствую заводом, вырабатывающим счастье|.

training_quotes = codecs.open(u'true_training.txt','r','utf-8').readlines() #0.688442211055 (137/199)
testing_quotes = codecs.open(u'true_testing.txt','r','utf-8').readlines()

def train_svm(mode):
    #0.917431192661 if nu == 0.05 | 0.923547400612 if nu == 0.1 | 0.909785932722 if nu == 0.3 | 0.802752293578 if nu == 0.8 | 0.93119266055 (609/654) if nu == 0.2
    #clf = svm.SVC(class_weight='auto', cache_size = 2000, probability=True)#, verbose = True, shrinking = False) #0.717058222676 (702/979) #no prob 0.70684371808
    if mode == 'quote':
        clf = svm.SVC(cache_size = 2000, probability=True) #prob 0.755873340143 (740/979) #no prob 0.742594484168
    else:
        clf = svm.SVC(cache_size = 2000, probability=True,kernel = 'linear')#distance: no prob 0.689479060266 prob 0.688457609806


    #TRAINING DATA
    X,y = makeMatrix(training_quotes,mode)
    clf = clf.fit(X,y)
#    pickle.dump(clf, open(mode+'_classifier.pickle','wb'))

    #TESTING DATA

    X_test, correct = makeMatrix(testing_quotes,mode)
    prediction = clf.predict(X_test)
    score = sum([int(prediction[i] == int(correct[i])) for i in range(len(correct))])
    print(mode+' classifier accuracy: {} ({}/{})'.format(float(score)/len(correct), score, len(correct)))

FEATNUM = 10

def train_trees(mode):
#    clf = ExtraTreesClassifier()
    X,y = makeMatrix(training_quotes,mode)

#    forest = ExtraTreesClassifier(n_estimators=FEATNUM,random_state=0)

#    forest = tree.DecisionTreeClassifier(n_estimators=FEATNUM, random_state=0)


    forest = RandomForestClassifier(n_estimators=FEATNUM, random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1][:FEATNUM]
    # Print the feature ranking
    print("Feature ranking:"+mode)
    marks = featuresNames()
    for f in range(FEATNUM):
        #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        print (str(f+1)+'. '+marks[int(indices[f])]+' '+str(importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(FEATNUM), importances[indices],
           color="b", yerr=std[indices], align="center")
    plt.xticks(range(FEATNUM), [marks[i] for i in indices])
    plt.xlim([-1, FEATNUM])
    plt.show()
"""
    dot_data = StringIO()
    tree.export_graphviz(forest, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #graph.write_pdf("classifier.pdf")
    graph.show()
"""


# == begin

train_svm("quote")
train_svm("borders")

train_trees("quote")
train_trees("borders")


"""
# get the separating hyperplane

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
"""


"""


print 'X.shape', X.shape
X_new = clf.fit(X, y).transform(X)
clf.feature_importances_
print clf.feature_importances_
print 'X_new.shape', X_new.shape
"""
