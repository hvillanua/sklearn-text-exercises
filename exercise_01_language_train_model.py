"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# The training data folder must be passed as first argument
languages_data_folder = sys.argv[1]
dataset = load_files(container_path=languages_data_folder, encoding='UTF-8')

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3), use_idf=False)

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clf = Pipeline([('vect', vectorizer),
                ('clf', Perceptron(n_jobs=-1)),
                ])
# TASK: Fit the pipeline on the training set
clf.fit(docs_train,y_train)
# TASK: Predict the outcome on the testing set in a variable named y_predicted
y_predicted = clf.predict(docs_test)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names))

# Plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

#import pylab as pl
#pl.matshow(cm, cmap=pl.cm.jet)
#pl.show()

# Predict the result on some short new sentences:
sentences = [
    u'This is a language detection test.',
    u'Ceci est un test de d\xe9tection de la langue.',
    u'Dies ist ein Test, um die Sprache zu erkennen.',
    u'Esto es un test de detecci\u00F3n de idioma.',
    u'\u3053\u308c\u306f\u3001\u8a00\u8a9e\u691c\u51fa\u30c6\u30b9\u30c8\u3067\u3059\u3002',
]
predicted = clf.predict(sentences)

for s, p in zip(sentences, predicted):
    print(u'The language of "%s" is "%s"' % (s, dataset.target_names[p]))
