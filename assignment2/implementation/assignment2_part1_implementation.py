


import re

from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10




with open("../../../data/assignment2/wv_50d.txt") as f:
    data_raw = f.readlines()




word_to_vector_dict = {arr.split(' ')[0] : [float(f.replace('\n', '')) for f in arr.split(' ')[1:]] for arr in data_raw}




def word_to_vector(word):
    try:
        return word_to_vector_dict[word]
    except KeyError:
        return np.zeros(50)




def return_input_data(type_):
    with open("../../../data/assignment2/senti_binary.{}".format(type_)) as f:
        sentences = [re.sub('-|\t',' ',t).replace('\n','').lower() for t in f.readlines()]
     
    splitted = [sentence.split(' ') for sentence in sentences]
    data_input = [(np.mean([word_to_vector(word) for word in arr[:-1] if not word == ""], axis=0),int(arr[-1])) for arr in splitted]
    
    return data_input

train_input = return_input_data('train')
test_input = return_input_data('test')




class Model(nn.Module):
    
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        
        super(Model, self).__init__()
        
        self.hl1 = nn.Linear(input_dim, hidden1_dim)
        self.hl1a = nn.Tanh()
        self.layer1 = [self.hl1, self.hl1a]
        
        self.hl2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hl2a = nn.Tanh()
        self.layer2 = [self.hl2, self.hl2a]
        
        self.ol = nn.Linear(hidden2_dim, output_dim)
        self.ola = nn.Softmax()
        self.layer3 = [self.ol, self.ola]
        
        self.layers = [self.layer1, self.layer2, self.layer3]
        
    def forward(self, x):
        
        out = x
        
        for pa, a in self.layers:
            
            out = a(pa(out))
        
        return out




model = Model(50, 100, 100, 2)
model.double()




def return_acc(model, data):
    model_eval = model.eval()
    pred = []
    Y = []
    for i, (x,y) in enumerate(torch.utils.data.DataLoader(dataset=data,shuffle=True)):
        with torch.no_grad():
            output = model_eval(x)
        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    return(accuracy_score(Y, pred)* 100)




class Trainer():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=128, shuffle=True)
        
    def train(self, lr, ne):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        self.model.train()
        
        self.costs = []
        self.acc = {}
        self.acc['train'] = []
        self.acc['test'] = []
        
        for e in range(ne):
            
            print('training epoch %d / %d ...' %(e+1, ne))
            
            train_cost = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):

                inputs = Variable(inputs)
                targets = Variable(targets)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                train_cost += loss
                loss.backward()
                optimizer.step()
            
            
            self.costs.append(train_cost/len(train_input))
            print('cost: %f' %(self.costs[-1]))




trainer = Trainer(model, train_input)




trainer.train(0.001, 500) 




train_acc = return_acc(trainer.model, train_input)
test_acc = return_acc(trainer.model, test_input)
print("Train acc: {:.3f}\nTest acc: {:.3f}".format(train_acc, test_acc))
with open("../results/assignment2_part1_results.txt", 'w') as f:
    f.write("{}\n{}".format(train_acc, test_acc))




plt.plot(range(len(trainer.costs)), trainer.costs)
plt.savefig('../plots/assignment2_part1_plots_loss.png')




path = "../trained_models/part1_state.chkpt"
torch.save(trainer, path)

