import nltk
import sklearn
import sklearn.feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from collections import defaultdict
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

nltk.download('stopwords')
nltk.download('punkt')
stopwords_english = stopwords.words('english')

# Import Data
data = pd.read_csv(r'news.csv')

# Stemmer


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


# Creates Bag of Words tf-idf representation, it's also normalized
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(data["text"])

X = tfs.toarray()
y = data["label"]

# Mapping the lables from REAL and FAKE to 1 and 0
y = y.map({'REAL': 1, 'FAKE': 0}).astype(int)


# Function to return a string containing the accuracy, f1_score, precision, and recall. Uses the cross_validate method from sklearn
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)
    return f"accuracy: {results['test_accuracy'].mean()}\nf1_score: {results['test_f1'].mean()}\nprecision: {results['test_precision'].mean()}\nrecall: {results['test_recall'].mean()}"


# Naive Bayes
model = MultinomialNB()
decision_tree_result = cross_validation(model, X, y, 5)
print("Naive Bayes: ")
print(decision_tree_result, "\n")

# Logistic Regression
model2 = LogisticRegression()
decision_tree_result2 = cross_validation(model2, X, y, 5)
print("Logistic Regression: ")
print(decision_tree_result2, "\n")

# SVM
model = LinearSVC()
decision_tree_result = cross_validation(model, X, y, 5)
print("SVM: ")
print(decision_tree_result, "\n")

# Knn
model3 = KNeighborsClassifier()
decision_tree_result3 = cross_validation(model3, X, y, 5)
print("K-Nearest Neighbours: ")
print(decision_tree_result3, "\n")
