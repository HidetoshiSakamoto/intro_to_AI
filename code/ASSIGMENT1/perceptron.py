#!/usr/bin/env python
# coding: utf-8

# # Imports and auxiliary functions

# In[15]:


# imports

import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[16]:


# activation functions

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def ploss(yhat, y):
    return max(0, -yhat*y)

def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))

# prediction functions

def ppredict(self, x):
    return self(x)

def lrpredict(self, x):
    return 1 if self(x)>0.5 else 0

# extra

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


# # Neuron and Trainer class definitions

# In[17]:


class Neuron:

    def __init__(self, dimension=1, weights=None, bias=None, activation=(lambda x: x), predict=(lambda x: x)):
    
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)
    
    def __str__(self):
        
        return "Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
    
    def __call__(self, x):
        
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat


# In[18]:


class Trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = ploss
        
    def cost(self, data):
        
        return np.mean([self.loss(self.model.predict(x), y) for x, y in data])
    
    def accuracy(self, data):
        
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
    
    def train(self, lr, ne):
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        for epoch in range(ne):
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))


# # Generating a toy dataset of two clusters of points (red and blue, labaled with -1 and 1 correspondingly)

# In[19]:


group1 = np.random.multivariate_normal(mean=[5, 5], cov=[[3, 0], [0, 3]], size=25)
group2 = np.random.multivariate_normal(mean=[15, 15], cov=[[3, 0], [0, 3]], size=25)


# In[20]:


plt.scatter([x for x, y in group1], [y for x, y in group1], color='r')
plt.scatter([x for x, y in group2], [y for x, y in group2], color='b')


# In[21]:


data = [(list(d), -1) for d in group1]+[(list(d), 1) for d in group2]
random.shuffle(data)


# # Initialize a neuron

# In[24]:


model = Neuron(dimension=2, activation=perceptron, predict=ppredict)


# In[25]:


print(model)


# In[26]:


# we use this to visualize the decision boundary before and after training
def draw_decision_boundary(dataset, model):
    weights = [model.b] + list(model.w)
    dataset = [d[0]+[d[1]] for d in dataset]
    plt.scatter([d[0] for d in dataset if d[2] == -1], [d[1] for d in dataset if d[2] == -1], c='red')
    plt.scatter([d[0] for d in dataset if d[2] == 1], [d[1] for d in dataset if d[2] == 1], c='blue')
    xmin, xmax = min([d[0] for d in dataset]), max([d[0] for d in dataset])
    ymin, ymax = min([d[1] for d in dataset]), max([d[1] for d in dataset])
    xscale = 1.25
    yscale = 1.25
    xs = np.linspace(xmin, xmax, 100)
    plt.plot(xs, [-weights[0]/weights[2] - weights[1]/weights[2]*x for x in xs], c='black')
    axes = plt.gca()
    axes.set_xlim([((xmin+xmax)-(xmax-xmin)*xscale)/2.0, ((xmin+xmax)+(xmax-xmin)*xscale)/2.0])
    axes.set_ylim([((ymin+ymax)-(ymax-ymin)*yscale)/2.0, ((ymin+ymax)+(ymax-ymin)*yscale)/2.0])
    plt.show()

# initial boundary

draw_decision_boundary(data, model)


# In[27]:


trainer = Trainer(data, model)


# In[28]:


trainer.accuracy(data)


# # Train the model

# In[29]:


trainer.train(0.01, 25)


# In[30]:


# final boundary

draw_decision_boundary(data, model)


# In[31]:


data


# In[ ]:




