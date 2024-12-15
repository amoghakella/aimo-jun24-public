import sys
import re
import subprocess
from time import time
from configs import TIMEOUT_CODE
from pdb import set_trace as bp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from subj_data import get_geo, get_alg, get_nt, get_cp
tf.autograph.set_verbosity(0)

global subj_model
global final

def train_subj():
  global subj_model
  global final

  geo1 = get_geo()

  alg1 = get_alg()
  
  nt1 = get_nt()
  
  cp1 = get_cp()
    
  alg = []
  cp = []
  geo = []
  nt = []

  maxrange = min(len(alg1), len(cp1), len(geo1), len(nt1))
  for x in range(maxrange):
    alg.append(alg1[x])
    cp.append(cp1[x])
    geo.append(geo1[x])
    nt.append(nt1[x])
  #adding words and frequencies for each subject to a list
  wordlist = []
  freq = []
  for x in alg:
    words = x.split()
  for word in words:
    lowerword = word.lower()
    if lowerword in wordlist:
      index = wordlist.index(lowerword)
      freq[index][0] += 1
    else:
      wordlist.append(lowerword)
      freq.append([1, 0, 0, 0])
  for x in cp:
    words = x.split()
    for word in words:
      lowerword = word.lower()
      if lowerword in wordlist:
        index = wordlist.index(lowerword)
        freq[index][1] += 1
      else:
        wordlist.append(lowerword)
        freq.append([0, 1, 0, 0])
  for x in geo:
    words = x.split()
    for word in words:
      lowerword = word.lower()
      if lowerword in wordlist:
        index = wordlist.index(lowerword)
        freq[index][2] += 1
      else:
        wordlist.append(lowerword)
        freq.append([0, 0, 1, 0])
  for x in nt:
    words = x.split()
    for word in words:
      lowerword = word.lower()
      if lowerword in wordlist:
        index = wordlist.index(lowerword)
        freq[index][3] += 1
      else:
        wordlist.append(lowerword)
        freq.append([0, 0, 0, 1])

  #creating a big list of words and their frequencies
  total = []
  for x in range(len(wordlist)):
    total.append([wordlist[x], freq[x]])

  #normalizes word frequencies, and puts frequent words in a list
#number of problems in each subject is [237, 44, 40, 91]
  final = []
  for word in total:
    if word[1][0] + word[1][1] + word[1][2] + word[1][3] >= 5 and len(word[0]) > 3:
      final.append(word[0])

  training_alg = []
  for x in range(len(alg)):
    array = []
    for y in range(len(final)):
      if final[y] in alg[x]:
        array.append(1)
      else:
        array.append(0)
    training_alg.append(array)

  training_cp = []
  for x in range(len(cp)):
    array = []
    for y in range(len(final)):
      if final[y] in cp[x]:
        array.append(1)
      else:
        array.append(0)
    training_cp.append(array)

  training_geo = []
  for x in range(len(geo)):
    array = []
    for y in range(len(final)):
      if final[y] in geo[x]:
        array.append(1)
      else:
        array.append(0)
    training_geo.append(array)

  training_nt = []
  for x in range(len(nt)):
    array = []
    for y in range(len(final)):
      if final[y] in nt[x]:
        array.append(1)
      else:
        array.append(0)
    training_nt.append(array)

  X = np.zeros(shape=(len(training_alg) + len(training_cp) + len(training_geo) + len(training_nt), len(final)))
  y = np.zeros(shape=(len(training_alg) + len(training_cp) + len(training_geo) + len(training_nt), 4))
  count = 0
  for element in training_alg:
    X[count] = element
    y[count] = [1, 0, 0, 0]
    count+=1
  for element in training_cp:
    X[count] = element
    y[count] = [0, 1, 0, 0]
    count+=1
  for element in training_geo:
    X[count] = element
    y[count] = [0, 0, 1, 0]
    count+=1
  for element in training_nt:
    X[count] = element
    y[count] = [0, 0, 0, 1]
    count+=1

  subj_model = Sequential(
      [               
          tf.keras.Input(shape=(len(final),)),
          tf.keras.layers.Dense(10, activation="sigmoid"),
          tf.keras.layers.Dense(4, activation="sigmoid")
      ], name = "my_model" 
  )

  [layer1, layer2] = subj_model.layers

  W1,b1 = layer1.get_weights()
  W2,b2 = layer2.get_weights()

  subj_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0005),
  )

  subj_model.fit(
    X,y,
    epochs=1000
  )

def test_subj(txt):
  X_input = np.zeros(shape=(1, len(final)))
  problem = txt.replace("\\", "").replace("'", "")
  delimiters = [",", "|", ";", "!", ".", "?"]

  for delimiter in delimiters:
    problem = " ".join(problem.split(delimiter))

  count = 0
  for element in final:
    if element in problem:
      X_input[0][count] = 1
    count+=1

  prediction = subj_model.predict(X_input)
  probalg = prediction[0][0]
  probcp = prediction[0][1]
  probgeo = prediction[0][2]
  probnt = prediction[0][3]
  if probalg > probcp and probalg > probgeo and probalg > probnt:
    return "A"
  if probcp > probalg and probcp > probgeo and probcp > probnt:
    return "C"
  if probgeo > probalg and probgeo > probcp and probgeo > probnt:
    return "G"
  if probnt > probalg and probnt > probcp and probnt > probgeo:
    return "N"
