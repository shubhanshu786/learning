
# coding: utf-8

# In[85]:


import tensorflow as tf


# In[86]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('', one_hot=True)


# In[87]:


train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels


# In[88]:


s = tf.InteractiveSession()


# In[89]:


feature_size = mnist.train.images.shape[1]
output_size = mnist.train.labels.shape[1]
h1_size = 392
h2_size = 196
h3_size = 98
h4_size = 49
h5_size = 24
X = tf.placeholder(tf.float32, shape=(None, feature_size))
Y = tf.placeholder(tf.float32, shape=(None, output_size))


# In[90]:


W = {
    'h1' : tf.Variable(tf.random_normal([feature_size, h1_size])),
    'h2' : tf.Variable(tf.random_normal([h1_size, h2_size])),
    'h3' : tf.Variable(tf.random_normal([h2_size, h3_size])),
    'h4' : tf.Variable(tf.random_normal([h3_size, h4_size])),
    'h5' : tf.Variable(tf.random_normal([h4_size, h5_size])),
    'out' : tf.Variable(tf.random_normal([h5_size, output_size]))
}
b = {
    'h1' : tf.Variable(tf.random_normal([h1_size])),
    'h2' : tf.Variable(tf.random_normal([h2_size])),
    'h3' : tf.Variable(tf.random_normal([h3_size])),
    'h4' : tf.Variable(tf.random_normal([h4_size])),
    'h5' : tf.Variable(tf.random_normal([h5_size])),
    'out' : tf.Variable(tf.random_normal([output_size]))
}


# In[91]:


#print(X.shape,X.dtype,Y.shape,Y.dtype, W['h1'].shape,W['h1'].dtype,
#      W['out'].shape,W['out'].dtype, b['h1'].shape,b['h1'].dtype,
#      b['out'].shape,b['out'].dtype)


# In[92]:


def model():
    hidden_layer1 = tf.sigmoid(tf.matmul(X, W['h1'])+b['h1'])
    hidden_layer2 = tf.sigmoid(tf.matmul(hidden_layer1, W['h2'])+b['h2'])
    hidden_layer3 = tf.sigmoid(tf.matmul(hidden_layer2, W['h3'])+b['h3'])
    hidden_layer4 = tf.sigmoid(tf.matmul(hidden_layer3, W['h4'])+b['h4'])
    hidden_layer5 = tf.sigmoid(tf.matmul(hidden_layer4, W['h5'])+b['h5'])
    output_layer = tf.nn.softmax(tf.matmul(hidden_layer5, W['out'])+b['out'])
    return output_layer


# In[93]:


model = model()
#print(model.shape, model.dtype)


# In[97]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
print(loss.shape, loss.dtype)


# In[98]:


#opt = tf.train.MomentumOptimizer(0.1,0.5).minimize(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model,1), tf.argmax(Y,1)), tf.float32))
#print(opt)


# In[99]:


s.run(tf.global_variables_initializer())


# In[100]:


for i in range(55000):
    batch_X, batch_Y = mnist.train.next_batch(50)
    s.run(opt, {X:batch_X, Y:batch_Y})
    if i%550 == 0:
        acc = s.run(accuracy, {X:test_X, Y:test_Y})
        print("Accuracy at %d-th Epoch : %f"%(i/550,acc*100))
print(s.run(accuracy, {X: test_X, Y:test_Y})*100)

