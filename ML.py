
# Import dependencies for pytorch
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import threading
import itertools
import time



# Function for loading symbol
def loading_symbol():
    symbols = itertools.cycle('\\|/-')  # Rotating line
    while not thread_stop_event.is_set():
        print('\r' + next(symbols), end='', flush=True)
        time.sleep(0.1)  # Adjust this to change the speed of rotation



# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

#create nueral network 
#testing

class imgclassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 32, (3,3)),  #nn layer 1
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3)), #nn layer 2
                nn.ReLU(),
                nn.Conv2d(64, 64, (3,3)), #nn layer 3
                nn.ReLU(),
                nn.Flatten(), 
                nn.Linear(64*(28-6)*(28-6), 10)  
        )
    
    def forward(self, x):
        return self.model(x)


#instance of nn

clf = imgclassification().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3) #adam does the calculus
lossfn = nn.CrossEntropyLoss() 

if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f)) #loads weights into clasifier

    for epoch in range(10): #train for 10 epochs
        correct = 0
        total = 0
        for batch in dataset:
            x, y = batch
            x, y = x.to("cpu"), y.to("cpu")
            yhat = clf(x)
            loss = lossfn(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            _, predicted = torch.max(yhat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {correct / total * 100}%") #epoch is the cycle of an entire dataset passed through the neural network

    thread_stop_event.set()
    load_thread.join()

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)

    img = Image.open('./Testingimages/985.jpg') #call jpeg file here 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu') #convert to a tensor
    print("The predicted result is" , (torch.argmax(clf(img_tensor)))) #return number image


