from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        # punctuation because word_tokenize outputs [..., 'word', '.']
        self.stopwords = stopwords.words('english') + list(punctuation)
    def __repr__(self):
        return 'LemmaTokenizer'
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.stopwords]

class StemTokenizer:
    def __init__(self, stemmer='lancaster'):
        if stemmer == 'lancaster':
            self.stemmer = LancasterStemmer()
        elif stemmer == 'snowball':
            self.stemmer = SnowballStemmer('english')
        elif stemmer == 'porter':
            self.stemmer = PorterStemmer()
        # we need to stem stopwords as well. stemming of puntuation is puntuation itself
        self.stopwords = [self.stemmer.stem(word) for word in stopwords.words('english')] + list(punctuation)
    def __repr__(self):
        return 'StemTokenizer(stemmer="' + self.stemmer + '"")'
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) if t not in self.stopwords]