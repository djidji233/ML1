%tensorflow_version 1.x
%matplotlib inline
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D

tf.reset_default_graph()
# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

colors=['y','c','g','m','r','b'] #za lepsi prikaz krivih na grafu
loss_array = [None for br in range (1,8)] # iz nekog razloga range (1,7) ne radi za nb_features=6

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

for nb_features in range(1,7):

  # Korak 1: Učitavanje i obrada podataka.
  filename = '/content/corona.csv'
  all_data = np.loadtxt(filename, delimiter=',', skiprows=0)
  data = dict()
  data['x'] = all_data[:, 0]
  data['y'] = all_data[:, 1]

  # Nasumično mešanje.
  nb_samples = data['x'].shape[0]
  indices = np.random.permutation(nb_samples)
  data['x'] = data['x'][indices]
  data['y'] = data['y'][indices]

  # Normalizacija (obratiti pažnju na axis=0).
  data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
  data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

  
  # Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma
  print('FEATURES = ', nb_features)
  # Kreiranje feature matrice.
  data['x'] = create_feature_matrix(data['x'], nb_features)

  # Iscrtavanje.
  plt.scatter(data['x'][:, 0], data['y'])
  plt.xlabel('X')
  plt.ylabel('Y')


  # Korak 2: Model.
  # Primetiti 'None' u atributu shape placeholdera i -1 u 'tf.reshape'.
  X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
  Y = tf.placeholder(shape=(None), dtype=tf.float32)
  w = tf.Variable(tf.zeros(nb_features))
  bias = tf.Variable(0.0)

  w_col = tf.reshape(w, (nb_features, 1))
  hyp = tf.add(tf.matmul(X, w_col), bias)

  # Korak 3: Funkcija troška i optimizacija.
  Y_col = tf.reshape(Y, (-1, 1))

  # Regularizacija
  lmbd = 0.01
  l2_reg = lmbd * tf.reduce_mean(tf.square(w))
  l1_reg = lmbd * tf.reduce_mean(tf.abs(w))

  mse = tf.reduce_mean(tf.square(hyp - Y_col))
  loss = tf.add(mse, l2_reg)

  # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa 
  # slozenijim funkcijama.
  opt_op = tf.train.AdamOptimizer().minimize(loss)

  # Korak 4: Trening.
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Izvršavamo 100 epoha treninga.
    nb_epochs = 100
    total_loss = 0
    for epoch in range(nb_epochs):
      
      # Stochastic Gradient Descent.
      epoch_loss = 0
      
      for sample in range(nb_samples):
        feed = {X: data['x'][sample].reshape((1, nb_features)), Y: data['y'][sample]}
        _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
        epoch_loss += curr_loss
          
      # U svakoj desetoj epohi ispisujemo prosečan loss.
      epoch_loss /= nb_samples # epoch_loss = average loss te epohe
      total_loss += epoch_loss # total_loss = svi lossovi za taj nb_features
      if (epoch + 1) % 100 == 0:
        print('Epoch: {}/{} | Avg loss: {:.5f}'.format(epoch+1, nb_epochs,epoch_loss))
        loss_array[nb_features] = total_loss # jer indeksiranje ide od 0
        print('Total loss =',total_loss)

    
    # Ispisujemo i plotujemo finalnu vrednost parametara.
    w_val = sess.run(w)
    bias_val = sess.run(bias)
    print('w = ', w_val, 'bias = ', bias_val)
    print('-------------------------------------------------')
    xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
    hyp_val = sess.run(hyp, feed_dict={X: xs})  # Bez Y jer nije potrebno.
    plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=colors[nb_features-1])
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

# prikaz svih 6 regresionih krivih i svih podataka
plt.show() 
# prikaz zavisnosti finalne funkcije troška od stepena polinoma (na celom skupu)
plt.plot(loss_array)
plt.xlabel('stepen polinoma')
plt.ylabel('total loss')
plt.show()
# Sta primecujemo?
# Primecujemo kako sto vise raste nb_features tj stepen polinoma nase funkcije
# to ona postaje "pametnija" i sa stepenom 5 ima najmanji loss.
# Medjutim, kako stepen poraste na 6, komplikujemo vise nego sto je potrebno
# overfittujemo pa loss ponovo raste.