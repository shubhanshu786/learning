
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


mnist = input_data.read_data_sets('',one_hot=True)


# In[20]:


x_train = mnist.train.images.reshape(-1,28,28,1)
y_train = mnist.train.labels
x_test = mnist.test.images.reshape(-1,28,28,1)
y_test = mnist.test.labels
x_val = mnist.validation.images.reshape(-1,28,28,1)
y_val = mnist.validation.labels
keep_probe = 0.5
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[21]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import Activation


# In[22]:


def make_model():
    #Model initialization
    model = Sequential()
    #First convolution layer
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding="same", input_shape=(28,28,1)))
    model.add(Activation('relu'))
    
    #First pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #Second convolution layer
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same"))
    model.add(Activation('relu'))
    
    #Second pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #Data reshaping/flattening
    model.add(Flatten())
    
    #First dense layer
    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    #Dropout to handle excessive model parameter
    model.add(Dropout(keep_probe))
    
    #Final Softmax layer 
    model.add(Dense(10, activation='softmax'))
    
    return model
    


# In[23]:


#model creation
model = make_model()


# In[24]:


#Model learning with ADAM optimizer, categorical_crossentropy loss
#accuracy as learning metrics
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])


# In[25]:


#Model detail
model.summary()


# In[27]:


#Model training with 100 epochs, batch size = 50
model.fit(x_train, y_train, 
          validation_data=(x_val, y_val), 
          epochs=5, 
          batch_size=50,
          shuffle=True)


# In[28]:


model.evaluate(x_test, y_test)

