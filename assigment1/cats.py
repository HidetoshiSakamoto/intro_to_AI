################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################

import numpy as np
import pickle as pkl
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lrpredict(self, x): 
    return 1 if self(x)>0.5 else 0

class Cat_Model:

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=None):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):
        info = "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
        return info

    def __call__(self, x):
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat

    def load_model(self, file_path):
        with open(file_path, mode='rb') as f:
            saved_model = pkl.load(f)
        return saved_model

    def save_model(self, file_path):
        with open(file_path, mode='wb') as f:
            pkl.dump(self, f)

class Cat_Trainer:
        
    def lrloss(yhat, y):
        return -1 if z<=0 else 1   
    
    def __init__(self,model, loss = lrloss):
        self.model = model
        self.loss = loss

    def accuracy(self, data):
        acc = 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
        return acc

    def train(self, lr, ne):
        accuracy = self.accuracy(Cat_Data())
        print("initial accuracy: %.3f" % (accuracy))
        
        for epoch in range(ne):
            data_tmp = Cat_Data()
            data_tmp._shuffle()
            for d in data_tmp:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(Cat_Data())
            if (epoch+1) % 100 == 0:
                print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(Cat_Data(type_="test"))))

class Cat_Data:

    def __init__(self, type_="train",relative_path='../../data/assignment1/', data_file_name='cat_data.pkl'):
        with open("{}/{}".format(relative_path, data_file_name), 'rb') as f:
            data_raw = pkl.load(f) 
        
        data = []
        for k, v in data_raw[type_].items():
            for arr in v:
                data.append((np.ravel(arr)/255, 1 if k == 'cat' else 0))
                
        self.index = 0
        self.data = data
        self.n_data = len(data)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.n_data - 1:
            raise StopIteration
        self.index += 1    
        return self.data[self.index]

    def _shuffle(self):
        self.data = np.random.permutation(self.data)

def main():

    model = Cat_Model(dimension=12288, activation=sigmoid, predict=lrpredict)   
    trainer = Cat_Trainer(model)
    trainer.train(1e-3, 10**3) 
    model.save_model('./saved_models/cat_model.pkl')

if __name__ == '__main__':
    main()
