#imports
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string


english_stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
punctuation.add("'s")


stemmer = PorterStemmer()

def stem(word):
    return stemmer.stem(word)

def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [stem(w) for w in words if w.lower() not in english_stopwords and w not in punctuation]
    return words



def bag_of_words(pattern_sentences,all_words):
    bag = [1 if w in pattern_sentences else 0 for w in all_words]
    return bag