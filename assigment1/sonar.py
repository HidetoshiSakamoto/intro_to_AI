import numpy as np
import pickle as pkl
import random
import os

def perceptron(z):
    return -1 if z<=0 else 1

def ppredict(self, x):
    return self(x)

class Sonar_Model:
    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict = (lambda x: x)):
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)

    def __str__(self):
        info = "Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
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


class Sonar_Trainer:
    
    def ploss(yhat, y):
        return max(0, -yhat*y)

    def __init__(self,model, loss = ploss):
        self.model = model
        self.loss = loss
        
    def accuracy(self, data):
        acc = 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
        return acc

    def train(self, lr, ne):
        accuracy = self.accuracy(Sonar_Data())
        print("initial accuracy: %.3f" % (accuracy))
        
        for epoch in range(ne):
            data_tmp = Sonar_Data()
            data_tmp._shuffle()
            for d in data_tmp:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(Sonar_Data())
            
            if (epoch + 1) % 100 == 0:
                print('>epoch=%d, learning_rate=%.4f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(Sonar_Data())))


class Sonar_Data:
    
    def __init__(self, relative_path='../../data/assignment1', data_file_name='sonar_data.pkl'):
        with open("{}/{}".format(relative_path, data_file_name), 'rb') as f:
            data_raw = pkl.load(f) 
        
        mu = np.ravel([f for i in list(data_raw.values()) for f in i]).mean()
        std = np.ravel([f for i in list(data_raw.values()) for f in i]).std()
    
        def standardize(x, mu, std):
            return (x - mu)/std
    
        data = []
        for k, v in data_raw.items():
            for arr in v:
                data.append((standardize(arr, mu, std), 1 if k == 'r' else -1))
        
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

    model = Sonar_Model(dimension=60, activation=perceptron, predict=ppredict)
    trainer = Sonar_Trainer(model)
    trainer.train(1e-4, 10**3)
    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')
    model.save_model('./saved_models/sonar_model.pkl')
    
if __name__ == '__main__':
    main()