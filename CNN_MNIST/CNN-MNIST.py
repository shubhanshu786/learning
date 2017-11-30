
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# In[10]:


mnist = input_data.read_data_sets('',one_hot=True)


# In[11]:


x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# In[12]:


def weight_variable(shape):
    vals = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(vals)


# In[13]:


def weight_bias(shape):
    vals = tf.constant(0.1, shape=shape)
    return tf.Variable(vals)


# In[14]:


def conv(X, W):
    return tf.nn.conv2d(X,W,strides=[1,1,1,1], padding='SAME')


# In[15]:


def pool(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='SAME')


# In[16]:


X = tf.placeholder(tf.float32, shape=(None,784))
Y = tf.placeholder(tf.float32,shape=(None,10))


# In[17]:


x_image = tf.reshape(X,[-1,28,28,1])


# In[18]:


print(x_image.shape)


# In[19]:


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = weight_bias([32])
h_conv1 = tf.nn.relu(conv(x_image,W_conv1)+b_conv1)
print(W_conv1.shape, b_conv1.shape, h_conv1.shape)


# In[20]:


h_pool1 = pool(h_conv1)
print(h_pool1.shape)


# In[21]:


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_bias([64])
h_conv2 = tf.nn.relu(conv(h_pool1,W_conv2)+b_conv2)
print(W_conv2.shape, b_conv2.shape, h_conv2.shape)


# In[22]:


h_pool2 = pool(h_conv2)
print(h_pool2.shape)


# In[23]:


W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = weight_bias([1024])


# In[24]:


h_pool2_flat = tf.reshape(h_pool2, shape=[-1,7*7*64])
print(h_pool2_flat.shape)


# In[25]:


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


# In[26]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[27]:


W_fc2 = weight_variable([1024,10])
b_fc2 = weight_bias([10])


# In[28]:


y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2


# In[29]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))


# In[30]:


regularizer = (tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(b_fc1)+
              tf.nn.l2_loss(W_fc2)+tf.nn.l2_loss(b_fc2))


# In[31]:


loss += 5e-4*regularizer


# In[34]:


batch = tf.Variable(0, tf.float32)
learning_rate = tf.train.exponential_decay(
    0.01,
    batch*100, #100 because of batch size
    x_train.shape[0],
    0.95,
    staircase=True
)


# In[35]:


opt = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,
                                                             global_step=batch)


# In[36]:


#opt = tf.train.AdamOptimizer(0.0001).minimize(loss)


# In[37]:


accuracy = tf.reduce_mean(tf.cast
                          (tf.equal
                           (tf.argmax(y_conv,1), 
                            tf.argmax(Y,1)),
                           tf.float32)
                         )


# In[38]:


s = tf.Session()


# In[39]:


s.run(tf.global_variables_initializer())


# In[40]:


for i in range(55000): #for 100 epochs
    batch = mnist.train.next_batch(100)
    s.run(opt,{X:batch[0],Y:batch[1],keep_prob:0.5})
    if i%200 == 0:
        acc = s.run(accuracy,{X:x_test,Y:y_test,keep_prob:1.0})*100
        print("Accuracy at %d-th Step : %f"%(i,acc))


# In[29]:


print(s.run(accuracy,{X:mnist.test.images, Y:mnist.test.labels, keep_prob:1.0}))

