
#applying taught material from Jose Portilla's course

import numpy as np 
import pandas as pd 
from IPython.display import Markdown, display 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

data = pd.read_csv('moviereviews.tsv', sep='\t')
print(data.head())

print(len(data))

display(Markdown('> '+data['review'][0]))


#======Removing null / empty data 

##check and remove null values
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

#removing empty strings from dataset
blank_spaces = []
for index, label, review in data.itertuples(): 
    if type(review) == str: 
            if review.isspace(): 
                blank_spaces.append(index)

print(len(blank_spaces), 'blanks: ', blank_spaces)
data.drop(blank_spaces, inplace=True)

print(data['label'].value_counts())

X = data['review']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#piplines for vectorising data

# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

#first pipeline
print('======First Pipeline: naive Bayes')
text_clf_nb.fit(X_train, y_train)
predictions = text_clf_nb.predict(X_test)
print('conf matrix:', metrics.confusion_matrix(y_test,predictions))
print('class report: ', metrics.classification_report(y_test,predictions))
print('accuracy score:',metrics.accuracy_score(y_test,predictions))

#second pipeline
print('========Second pipeline: linear svc')
text_clf_lsvc.fit(X_train, y_train)
predictions = text_clf_lsvc.predict(X_test)
print('conf matrix:', metrics.confusion_matrix(y_test,predictions))
print('class report: ', metrics.classification_report(y_test,predictions))
print('accuracy score:',metrics.accuracy_score(y_test,predictions))

#narrowing stop words to see effect on accuracy score 
stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

text_clf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),
                     ('clf', LinearSVC()),
])
text_clf_lsvc2.fit(X_train, y_train)

print('==========repeating with smaller stop word pool (with lin svc)')
predictions = text_clf_lsvc2.predict(X_test)
print('conf matrix:', metrics.confusion_matrix(y_test,predictions))
print('class report: ', metrics.classification_report(y_test,predictions))
print('accuracy score:',metrics.accuracy_score(y_test,predictions))

with open('my_review.txt') as f:
    my_review  = f.read()

print(text_clf_nb.predict([my_review]))
print(text_clf_lsvc.predict([my_review]))
