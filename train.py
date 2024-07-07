import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocessing import *

with open ('ECchatBotData.json','r') as f:
    intents = json.load(f) 

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

all_words = sorted(set(all_words))
tags = sorted(tags)


x_train = []
y_train = []

for (pattern_sentences,tag) in xy:
    bag = bag_of_words(pattern_sentences,all_words)
    bag = np.array(bag,dtype=np.float32)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#Hyperparameters
batch_size = 8
input_size = len(all_words)
hidden_size = 8
num_classes = len(tags)
learning_rate = 0.001
num_epochs = 1000 

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,batch_size = batch_size, shuffle = True)


class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size,hidden_size, num_classes).to(device)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            
            #forward
            outputs = model(words)
            loss = criterion(outputs,labels.long())
            
            #backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if(epoch+1) % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, loss = {loss.item():.9f}")
    print(f"final loss = {loss.item():.9f}")


    data = {
        "model_state":model.state_dict(),
        "input_size":input_size,
        "hidden_size":hidden_size,
        "output_size":num_classes,
        "all_words":all_words,
        "tags":tags
    }

    file = "chatbotdata.pth"
    torch.save(data,file)
                                                            
    print(f"Training data is saved in {file}")


