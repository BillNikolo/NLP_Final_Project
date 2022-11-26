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

nltk.download('stopwords')
stopwords_english = stopwords.words('english')

data = pd.read_csv(r'news.csv')
text_column = pd.DataFrame(data, columns=['text']).values.tolist()
labels = pd.DataFrame(data, columns=['label']).values.tolist()
df = pd.DataFrame(data, columns=['text', 'label']).values.tolist()
# print(text)
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

clean_words = []
stemmer = PorterStemmer()
words_stem = []
# extra_punctuation = ['—', '...', '’', '“', '‹', '›', '…']
for _ in text_column:
    _[0] = re.sub(r'https?:\/\/.*[\r\n]*', '', _[0])
    _[0] = re.sub(r'#', '', _[0])
    _[0] = tokenizer.tokenize(_[0])
    c_w = []
    s_w = []
    for word in _[0]:
        if word not in stopwords_english and word not in string.punctuation:
            c_w.append(word)
    for w in c_w:
        s_w.append(stemmer.stem(w))
    clean_words.append(c_w)
    words_stem.append(s_w)

print(words_stem)
