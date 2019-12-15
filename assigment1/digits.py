import numpy as np
import pickle as pkl
import random
import os

def affine(z, W, b):
    return np.dot(z, W) + b

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(u):
    max_u = np.max(u, axis=1, keepdims=True)
    exp_u = np.exp(u-max_u)
    return exp_u/np.sum(exp_u, axis=1, keepdims=True)

def make_one_hot(d):
    return np.eye(10)[d]

class Digits_Model:
    def __init__(self, n_data=None, dim_input=None, dim_hidden=None, dim_out=None, weights1=None, weights2=None, bias1=None, bias2=None,                 activation1=(lambda x: x), activation2=(lambda x: x)):
        self.n_data = n_data
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        
        self.W1 = np.random.normal(size=(dim_input, dim_hidden))
        self.b1 = np.random.normal(size=(n_data, dim_hidden))
        self.W2 = np.random.normal(size=(dim_hidden, dim_out))
        self.b2 = np.random.normal(size=((n_data, dim_out)))
        self._a1 = activation1
        self._a2 = activation2
        
    def __str__(self):
        info = "Single hidden layer nn\n        \tInput dimension: %d\n        \tHidden layer dimension: %d\n        \tOutput layer dimension: %d\n        \tActivation1: %s\n        \tActivation2: %s" % (self.dim_input, self.dim_hidden, self.dim_out,self._a1.__name__, self._a2.__name__)
        return info
    
    def affine(z, W, b):
        return np.dot(z, W) + b

    def __call__(self, X0):
        Z1 = affine(X0, self.W1, self.b1)
        X1 = sigmoid(Z1)
        Z2 = affine(X1, self.W2, self.b2)
        yhat = softmax(Z2)
        return yhat

    def predict(self, X0):
        yhat = self(X0)
        d = np.argmax(yhat, axis=1)
        return d

    def load_model(self, file_path):
        with open(file_path, mode='rb') as f:
            saved_model = pkl.load(f)
        return saved_model

    def save_model(self, file_path):
        with open(file_path, mode='wb') as f:
            pkl.dump(self, f)

class Digits_Data:

    def __init__(self, relative_path='../../data/assignment1/', data_file_name='digits_data.pkl', batch_size=None):
        with open("{}/{}".format(relative_path, data_file_name), 'rb') as f:
            data_raw = pkl.load(f) 
            
        data = []
        type_ = 'train'
        for k, v in data_raw[type_].items():
            for arr in v:
                data.append((np.ravel(arr)/255, make_one_hot(k)))
        
        
        self.index = -1
        self.data = np.array(data)
        self.n_data = self.data.shape[0]
        
    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1   
        if self.index == self.n_data:
            raise StopIteration 
        return self.data[self.index]

    def _shuffle(self):
        self.data = np.random.permutation(self.data)

class Digits_Trainer:

    def __init__(self, dataset, model):
        data = [d for d in dataset]
        self.X0 = np.array([d[0] for d in data])
        self.y = np.array([d[1] for d in data])
        self.model = model

    def accuracy(self):
        max_yhat = self.model.predict(self.X0)
        max_y = np.argmax(self.y, axis=1)
        return 100*np.sum(max_yhat == max_y)/len(self.y)
    def forward(self, X0):
        Z1 = affine(X0, self.model.W1, self.model.b1)
        X1 = sigmoid(Z1)
        Z2 = affine(X1, self.model.W2, self.model.b2)
        yhat = softmax(Z2)
        return Z1, X1, Z2, yhat
    def cross_entropy_loss(self, yhat, y):
        return -np.sum(y * np.log(yhat))
    def dsigmoid(self, x):
        return (1.0 - sigmoid(x)) * sigmoid(x)    
    def calc_delta_hidden(self, Z, D, W):
        return self.dsigmoid(Z) * np.dot(D, W)
    def softmax_cross_entropy_error_back(self, yhat, y):
        return (yhat - y)/y.shape[0]
    def dweight(self, D, X):
        return np.dot(D.T, X)
    def dbias(self, D):
        return D.sum(axis=0)          
    def backward(self, yhat, y, X1, W2):
        d_output = self.softmax_cross_entropy_error_back(yhat,y)
        d_hidden = self.calc_delta_hidden(X1,d_output, W2.T)
        return d_output, d_hidden
    def update_params(self, delta_output, delta_hidden, X1, X0, lr):
        W2 = self.model.W2 - lr * self.dweight(X1, delta_output)
        W1 = self.model.W1 - lr * self.dweight(X0, delta_hidden)
        b2 = self.model.b2 - lr * self.dbias(delta_output)
        b1 = self.model.b1 - lr * self.dbias(delta_hidden)
        return W2, W1, b2, b1
    def train(self, lr, ne):
        accuracy = self.accuracy()
        print("initial accuracy: %.3f" % (accuracy))
        for i in range(ne):
            Z1, X1, Z2, yhat = self.forward(self.X0)
            delta_output, delta_hidden = self.backward(yhat,self.y, X1, self.model.W2)
            self.model.W2, self.model.W1, self.model.b2, self.model.b1= self.update_params(delta_output, delta_hidden, X1,self.X0, lr)
            if (i+1) % 100 == 0:
                accuracy = self.accuracy()
                print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (i+1, lr, accuracy))
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy()))


def main():

    data = Digits_Data()
    model = Digits_Model(n_data=data.data.shape[0], dim_input=data.data[0][0].shape[0], dim_hidden=100, dim_out=data.data[0][1].shape[0], activation1=sigmoid, activation2=softmax)
    trainer = Digits_Trainer(data, model)
    trainer.train(0.5, 10**3)
    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')
    model.save_model('./saved_models/digits_model.pkl')

if __name__ == '__main__':

    main()

