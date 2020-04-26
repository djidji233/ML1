%tensorflow_version 1.x
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib inline

class KNN:
  
  def __init__(self, nb_features, nb_classes, data, k, weighted = False):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.data = data
    self.k = k
    self.weight = weighted
    
    # Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
    self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Q = tf.placeholder(shape=(nb_features), dtype=tf.float32)
    
    # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), 
                                  axis=1))
    _, idxs = tf.nn.top_k(-dists, self.k)  
    
    self.classes = tf.gather(self.Y, idxs)
    self.dists = tf.gather(dists, idxs)
    
    if weighted:
       self.w = 1 / self.dists  # Paziti na deljenje sa nulom.
    else:
       self.w = tf.fill([k], 1/k)
    
    # Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)
    
    # Klasa sa najvise glasova je hipoteza.
    self.hyp = tf.argmax(self.scores)
  
  # Ako imamo odgovore za upit racunamo i accuracy.
  def predict(self, query_data):
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
     
      nb_queries = query_data['x'].shape[0]
      
      matches = 0
      for i in range(nb_queries):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'], 
                                                  self.Y: self.data['y'], 
                                                 self.Q: query_data['x'][i]})
        if query_data['y'] is not None:
          actual = query_data['y'][i]
          match = (hyp_val == actual)
          if match:
            matches += 1
          # if i % 10 == 0:
          #   print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'.format(i+1, nb_queries, hyp_val, actual, match))
    
      # print('{} matches out of {} examples'.format(matches, nb_queries))
      return matches / nb_queries

filename = 'Prostate_Cancer.csv'
import random


#Loading data
results = np.loadtxt(filename, delimiter=',', dtype=str, skiprows=1, usecols=(1))
all_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(2, 3, 4, 5))


#Mixing data
mixing_data = True
if mixing_data == True:
  rand = random.randrange(1,100)
  all_data = shuffle(all_data, random_state=rand)
  results = shuffle(results, random_state=rand)


#Spliting data
train_count = round(len(all_data)*0.8)
train_x = all_data[:train_count,:]
test_x = all_data[train_count:,:]
train_y = results[:train_count].tolist()
test_y = results[train_count:].tolist()


#Replacing 'M' & 'B' to '1' & '0'
for letter in train_y:
  if letter == 'M': train_y.append(1)
  if letter == 'B': train_y.append(0)
train_y = train_y[int(len(train_y)/2):]

for letter in test_y:
  if letter == 'M': test_y.append(1)
  if letter == 'B': test_y.append(0)
test_y = test_y[int(len(test_y)/2):]

train_y = np.asarray(train_y)
test_y = np.asarray(test_y)


#Reshaping data
nb_train = len(train_y)
nb_test = len(test_y)
train_x = np.reshape(train_x, [nb_train, -1])
test_x = np.reshape(test_x, [nb_test, -1])


#Sending data
res = []
nb_features = 4
nb_classes = 2
for i in range(1,16):
  train_data = {'x': train_x, 'y': train_y}
  knn = KNN(nb_features, nb_classes, train_data, i, weighted = False)
  accuracy = knn.predict({'x': test_x, 'y': test_y})
  print(str(i) +'th Test set accuracy: ', accuracy)
  res.append(accuracy)


#Drawing data
plt.plot(np.arange(1,16), res)


#Evaluacija rezultata
# Kako radimo na veoma malom setu podataka ukljucivanjem mesanja dobijamo razlicite rezultate.
# Ipak cini nam se da je najveci % kada je k izmedju 3 - 6, pa zatim opada, dok opet ne dodje do 
# dobrih rezultata pri kraju, tj sa vrednostima k izmedju 13 - 15, ali opet ne i najboljim.
# Neki zakljucak bi bio da vecim k ne dobijamo bolje rezultate, mozda cak i da je kontra-efektivno.