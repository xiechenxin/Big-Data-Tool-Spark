# Databricks notebook source
# DBTITLE 1,A3: Deep Learning
#This assignment is a guided notebook for Assignment 3 on Deep Learning.
#You can already complete the exercises in this notebook after the first session on Deep Learning.

# COMMAND ----------

# DBTITLE 1,Q1: Single-label Multiclass Classification
#This exercise introduces multiclass classification using Deep Learning.

# COMMAND ----------

#Load the dataset
#################

from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#The dataset used here contains Reuters newswires that should be classified into 46 mutually exclusive topics.

# COMMAND ----------

#Prepare data
#############

#x
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# COMMAND ----------

#y
from keras.utils.np_utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

#Alternative: encode the labels by casting them to an integer tensor
#y_train = np.array(train_labels)
#y_test = np.array(test_labels)
#Remark: in this case, you need to choose sparse_categorical_crossentropy as the loss function 
#since categorical_crossentropy expects the labels to follow a categorical encoding

# COMMAND ----------

#Split the training dataset in a training and validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# COMMAND ----------

#Exercise 1A: Estimate a model with 3 layers and 1 output layer. Determine the number of hidden units.
#Use the same number of units for the first 3 layers.
#For these first 3 layers, use a number of units that is larger than the number of output classes but smaller than 100 (taking into account the 2^n rule).

#Instead of 16 hidden units to learn 46 classes, use 64 units. 16 units is too restrictive: every layer could serve as a bottleneck
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax')) #output a probability distribution over the 46 different output classes.
#You end up with a network with a Dense layer of size 46

# COMMAND ----------

#Define an optimizer, loss function, and metric for success
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

#Fit the model. Use batch_size=512 and epoch=20 as start values.
history = model.fit(partial_x_train,
  partial_y_train,
  epochs=20,
  batch_size=512,
  validation_data=(x_val, y_val))

# COMMAND ----------

#At which value for epoch does the network overfit?
#Copy-paste the answer in the comments.

#Network seems to overfit after 8 epochs (see graphs)
epoch_optim=8

# COMMAND ----------

#Look at results (accuracy)
import matplotlib.pyplot as plt
plt.clf()

#Create values
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
epochs = range(1, len(acc_values) + 1)

#Create plot
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# COMMAND ----------

#Retrain the model with the optimal number of epochs. What is the accuracy?
#Copy-paste the answer in the comments.

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=epoch_optim, batch_size=512)

results_optim_epoch = model.evaluate(x_test, y_test)

# COMMAND ----------

results_optim_epoch

# COMMAND ----------

#Exercise 1B: Estimate the model in Exercise 1 again but with a batch_size=32. What can you conclude in terms of accuracy?
#Copy-paste the accuracy and the answer in the comments.
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=epoch_optim, batch_size=32)

results_batchsize = model.evaluate(x_test, y_test)

# COMMAND ----------

results_batchsize
#The accuracy has increased.

# COMMAND ----------

#Exercise 2: Starting from the model in Exercise 1, estimate a model where the 1st layer is doubled in terms of units.
#Add an additional layer after the first layer where the number of units is also doubled.
#What can you conclude on the accuracy of the model compared to the model in Exercise 1?
#Copy-paste the accuracy of this model and your conclusion in the comments.

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=epoch_optim, batch_size=512)

results_exc2 = model.evaluate(x_test, y_test)

# COMMAND ----------

results_exc2

# COMMAND ----------

#Exercise 3: Starting from the model in Exercise 2, estimate a model where you add an additional layer of 64 units after the 2nd layer. 
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy you get from your model in the comments.

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=epoch_optim, batch_size=512)

results_exc3 = model.evaluate(x_test, y_test)

# COMMAND ----------

results_exc3

# COMMAND ----------

#Exercise 4: Starting from the model in Exercise 2, change the number of units to 4 for the 2nd layer.
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy you get from your model in the comments.

#This is a perfect example of an information bottleneck. Accuracy drops significantly by reducing the number of layers to 4.
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=epoch_optim, batch_size=512)

results_exc4 = model.evaluate(x_test, y_test)

# COMMAND ----------

results_exc4
