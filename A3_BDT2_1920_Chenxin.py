# Databricks notebook source
#Chenxin Xie

# COMMAND ----------

# DBTITLE 1,A3: Deep Learning
#This assignment is a guided notebook for Assignment 3 on Deep Learning.
#You can already complete the exercises in this notebook after the first session on Deep Learning.
#This exercise introduces single-label multiclass classification using Deep Learning. This is a popular technique for f.ex. image classification.

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

#Exercise 1A: Estimate a model with 3 layers and 1 output layer.
 #Determine the number of hidden units as follows:
  #Use the same number of units for the first 3 layers.
  #For these first 3 layers, use a number of units that is larger than the number of output classes but smaller than 100 (taking into account the 2^n rule).

# COMMAND ----------

from keras import models
from keras import layers
# define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation = "softmax"))

# COMMAND ----------

#Define an optimizer, loss function, and metric for success.
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

#Fit the model. Use batch_size=512 and epoch=20 as starting values.
history = model.fit(partial_x_train,
  partial_y_train,
  epochs=20,
  batch_size=512,
  validation_data=(x_val, y_val))

# COMMAND ----------

#Look at the results (accuracy)
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

#At which value for epoch does the network overfit?
#Copy-paste the answer in the comments.

# start from about epochs=10, the network is overfitting

# COMMAND ----------

#Retrain the model with the optimal number of epochs. What is the final accuracy?
#Copy-paste the answer in the comments.


# COMMAND ----------

model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=8, batch_size=512)

results_epoch_optim = model.evaluate(x_test, y_test)

# COMMAND ----------

results_epoch_optim

# COMMAND ----------

# Result for Exercise 1
  # retrain the model with epochs=8
  # final accurcy is 0.9905919910326454

# COMMAND ----------

#Exercise 1B: Fit the model in Exercise 1 again but with a batch_size=32. What can you conclude in terms of accuracy?
#Copy-paste the accuracy and your answer in the comments.


# COMMAND ----------

# model for Exercise 1B
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

model.fit(x_train, y_train, epochs=8, batch_size=32)

results_change_batchSize = model.evaluate(x_test, y_test)
results_change_batchSize

# Result for Exercise 1B
  # accuracy is 0.9863912880898371.
  # Conclusion is : accuracy dropped a bit. So decreasing batch_size would not improve accuracy.

# COMMAND ----------

#Exercise 2: Starting from the model in Exercise 1, estimate a model where the 1st layer is doubled in terms of units.
#Add an additional layer after the first layer where the number of units is also doubled.
#What can you conclude on the accuracy of the model compared to the model in Exercise 1?
#Copy-paste the accuracy of this model and your conclusion in the comments.


# COMMAND ----------

# model for Exercise 2
# define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation = "softmax"))

# define optimizer, loss and metrics of success 
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

# fit the model
model.fit(x_train, y_train, epochs=8, batch_size=32)

# check the accuracy
results_change_layer = model.evaluate(x_test, y_test)
results_change_layer


# Result for Exercise 2
  # the accuracy of this model is 0.9722599110641548.
  # conclusion is : accuracy dropped a bit after layer number and unit number of layer has been increased. So increasing number of units or adding layer would not improve accuracy.

# COMMAND ----------

#Exercise 3: Starting from the model in Exercise 2, estimate a model where you add an additional layer of 64 units after the 2nd layer. 
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy of this model and your conclusion in the comments.


# COMMAND ----------

# model for Exercise 3

# define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation = "softmax"))

# define optimizer, loss and metrics of success 
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

# fit the model
model.fit(x_train, y_train, epochs=8, batch_size=32)

# check the accuracy
results_add_layer = model.evaluate(x_test, y_test)
results_add_layer

# Result for Exercise 3
  # the accuracy of this model is 0.9782609340344172.
  # conclusion is : accuracy improved a bit, compared with excercise 2, after adding an additional layer of 64 units. So adding layers could improve accuracy.
  # Compared with exercise 1, the accuracy dropped a bit, so increasing units and adding layer at the same time, does not necessary for imporoving accuracy.

# COMMAND ----------

#Exercise 4: Starting from the model in Exercise 2, change the number of units to 4 for the 2nd layer.
#What can you conclude about the accuracy of the model compared to the model in Exercise 2?
#Copy-paste the accuracy of this model and your conclusion in the comments.


# COMMAND ----------

# model for Exercise 4
# define the model
model = models.Sequential()

#Define the layers
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation = "softmax"))

# define optimizer, loss and metrics of success 
model.compile(optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['acc'])

# fit the model
model.fit(x_train, y_train, epochs=8, batch_size=32)

# check the accuracy
results_change_layer_2 = model.evaluate(x_test, y_test)
results_change_layer_2


# Result for Exercise 4
  # the accuracy of this model is 0.9888110506354117.
  # conclusion is : accuracy improved a bit after decreasing units of one layer. Compared with exercise 2 and 3, this model has the highest accuracy.
  # It means adding layers or adding layers with smaller number would have a higher accuracy than adding layers with higher number.
