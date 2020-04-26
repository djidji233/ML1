%tensorflow_version 1.x
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib inline
from nltk.tokenize import regexp_tokenize, word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
import math
from spellchecker import SpellChecker
from nltk.corpus import brown
from nltk import FreqDist
from collections import Counter
import re, string
from re import search
from math import ceil
!pip install pyspellchecker
!pip install nltk
import nltk
nltk.download()

pattern1 = "http\S+|[a-z0-9.\-]+[.](?:com/?)[a-z0-9.\-]+|[a-z0-9.\-]+[.](?:com/?)"
pattern2 = r"'|@[\w]*|#|&|^\s+|\s+$|\d"
pattern3 = r"[^a-zA-Z0-9\s]+"

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
    nb_examples = X.shape[0]

    # Racunamo P(Klasa) - priors
    # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
    # broja u intervalu [0, maksimalni broj u listi]
    self.priors = np.bincount(Y) / nb_examples

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += cnt
    
    # Racunamo P(Rec_i|Klasa) - likelihoods
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
          
  def predict(self, bow, match):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob

    acc = 0
    prediction = np.argmax(probs)
    if prediction == match:
      acc = 1
    return (prediction,acc)

def tfidf_score(word, doc):
  tf = freq_score(word, doc)
  idf = idf_table[word]
  return tf * idf

def occ_score(word, doc):
   return 1 if word in doc else 0
  
def numocc_score(word, doc):
  return doc.count(word)

def freq_score(word, doc):
  if(len(doc) == 0): return 0
  return doc.count(word) / len(doc)

def has_num(s):
  return any(i.isdigit() for i in s)

def has_elong(sentence):
  elong = re.compile("([a-zA-Z])\\1{2,}")
  return bool(elong.search(sentence))

def check_elongated(tweet):
  clean = []
  for word in tweet.split():
    if has_elong(word):
      clean.append(re.sub(r'(?i)(.)\1+', r'\1', word))
    else:
      clean.append(word)
  return " ".join(clean)

def clean_tweet(tweet):
  clean = tweet.lower()
  clean = re.sub(pattern1, ' ', clean)
  clean = re.sub(pattern2, '', clean)
  clean = re.sub(pattern3, ' ', clean)
  clean = check_elongated(clean)
  return clean

def LR(word):
  if pos_vocab[word]>10  and neg_vocab[word]>10:
    return pos_vocab[word]/neg_vocab[word]
  return 0

def score(num):
  mul = 10**2
  return ceil(num * mul)/mul

filename = 'twitter.csv'

results = np.loadtxt(filename, delimiter=',', dtype=int, encoding='latin-1', skiprows=1, usecols=1)
all_data = np.loadtxt(filename, delimiter=',', dtype=str, encoding='latin-1', skiprows=1, usecols=2)


clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))
porter = PorterStemmer()
lancaster = LancasterStemmer()
spell = SpellChecker()
word_list = brown.words()
word_set = set(word_list)


for row in all_data:
  tweet = clean_tweet(row)
  words = regexp_tokenize(tweet, "[\w']+")
  words_filtered = [w for w in words if w not in stop_punc]
  words_stemmed = [porter.stem(w) for w in words_filtered]
  words_big = [w for w in words_stemmed if len(w) > 3]
  clean_corpus.append(words_big)


vocab_set = Counter()
pos_vocab = Counter()
neg_vocab = Counter()
lr = Counter()
row = 0

for doc in clean_corpus:
  if results[row] == 1:
    for word in doc:
      pos_vocab[word]+=1  
  elif results[row] == 0:
    for word in doc:
      neg_vocab[word]+=1  
  for word in doc:
    vocab_set[word]+=1
  row += 1

vocab = [word for word, word_count in vocab_set.most_common(10000)]

for w in vocab: lr[w] = LR(w)
lr += Counter()
for i in lr: lr[i] = round(lr[i] ,2)

print("Top positive   : ", pos_vocab.most_common(5))
print("Top negative   : ", neg_vocab.most_common(5))
print("Top LR         : ", lr.most_common(5))
print("Bot LR         : ", lr.most_common()[:-5-1:-1])

#BOW
X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
for doc_idx in range(len(clean_corpus)):
  doc = clean_corpus[doc_idx]
  for word_idx in range(len(vocab)):
    word = vocab[word_idx]
    cnt = occ_score(word, doc)
    X[doc_idx][word_idx] = cnt

mixing_data = False
if mixing_data == True:
  X = shuffle(X, random_state=30)
  results = shuffle(results, random_state=30)

#Spliting data
train_count = round(len(X)*0.8)
train_x = X[:train_count,:]
test_x = X[train_count:,:]
train_y = results[:train_count]
test_y = results[train_count:]

class_names = ['0','1']
matrix = np.zeros((2,2))

model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=10)
model.fit(train_x, train_y)

accuracy = 0
for i, test in enumerate(test_x):
  prediction, acc = model.predict(test,test_y[i])
  accuracy += acc
  if class_names[prediction] == '0' and test_y[i] == 0:
    matrix[0,0] += 1
  elif class_names[prediction] == '1' and test_y[i] == 0:
    matrix[0,1] += 1
  elif class_names[prediction] == '0' and test_y[i] == 1:
    matrix[1,0] += 1
  elif class_names[prediction] == '1' and test_y[i] == 1:
    matrix[1,1] += 1

print('Accuracy: ', score(accuracy/len(test_x)), "\n")
print("Confusion Matrix:")
print("True  negative:   ", matrix[0,0])
print("False positive:   ", matrix[0,1])
print("False negative:   ", matrix[1,0])
print("True  positive:   ", matrix[1,1])
print("\n", matrix)

#Evaluacija rezultata
# Sto se tice 5 najcesce koriscenih reci, primecujemo da se javljaju dve iste reci u oba skupa.
# Kako je to rec "like" verujemo da se javlja u toliko slucajeva jer moze imati i pozitivno i negativno
# znacenje, pa samim tim i ne doprinosi previse odredjivanju da li je tweet pozitivan ili ne.
# Sto se tice LR metrike, primecujemo da reci sa visim vrednostima oznacavaju reci u pozitivnim tvitovima,
# dok one sa nizim vrednostima sto blizim 0, oznacavaju reci u negativnim tvitovima. Reci koje imaju vrednosti
# blizu 1, ali ne padaju ispod, oznacavju reci koje nemaju veliki uticaj na to da li je tweet pozitivan ili ne.
# Kada pogledamo 10 reci sa top LR vrednostima i 10 reci koje se najvise ponavljaju, vidimo, da iako se LR reci,
# ne pojavljuju toliko, njihovo pojavljivanje u tweet-u doprinosi laksem odredjivanju klase tweet-a, nego reci koje
# su se najvise pojavljivale. Imaju vecu ulogu u odredjivanju klase tweet-a.