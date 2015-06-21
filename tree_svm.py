#!/usr/bin/env python
#! -*- coding: utf-8 -*-
from __future__ import division
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
from sklearn.svm import SVC
import os
import pdb
# import pydot
import random
import codecs
import pickle
import csv
import numpy as np


from sklearn.cluster import DBSCAN
from sklearn import metrics


def list_to_str(x):
    result = [str(y) for y in x]
    return '\t'.join(result)


def main():
#    lines = 'corpora_markers_distance_nonfiction_woNaN.csv',
#        'r','utf8').read().split('\n')

    features = [line.rstrip('\n').split('#') for line in codecs.open('corpora_markers_distance_nonfiction_left.csv','r').readlines()]
#    features = list(csv.reader(open('/Users/tysh/PycharmProjects/diploma/corpora_markers_distance_nonfiction_leftinside.csv','rb'),delimiter = '#'))

#    features = [dict(zip(lines[0],line)) for line in lines[1:]]

#    dv = DictVectorizer(sparse=False)
#    vectors = dv.fit_transform(features)
#    print dv.get_feature_names()

    subset = features[1:]
#    random.shuffle(subset)

    threshold = int(len(subset)/5*3)

    # clf = tree.DecisionTreeClassifier()
    clf = SVC(probability=True)

    # clf = RandomForestClassifier()
    X = np.array([feature[1:] for feature in subset[:threshold]])
    y = np.array([int(feature[0]) for feature in subset[:threshold]])
    print y
    clf = clf.fit(X,y)
    pickle.dump(clf, open('classifier.pickle','wb'))
    # clf = pickle.load(open('classifier.pickle','rb'))
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("classifier.pdf")

    correct = [feature[0] for feature in subset[threshold:]]
    prediction = clf.predict([feature[1:] for feature in subset[threshold:]])
    open('prediction.tsv','wb').write(list_to_str(prediction))
    open('correct.tsv','wb').write(list_to_str(correct))

    score = sum([int(prediction[i] == int(correct[i]))
        for i in range(len(correct))])
    print('tree classifier accuracy: {} ({}/{})'.format(
        score/len(correct), score, len(correct)))

    db = DBSCAN(eps=0.6, min_samples=4).fit(X) #dbscan = DBSCAN(random_state=111)

#    db = dbscan.fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
    pl.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2',
        'Noise'])
    pl.title('DBSCAN finds 2 clusters and noise')
    pl.show()

    ##############################################################################
    raw_input()
    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()



if __name__ == "__main__":
    main()