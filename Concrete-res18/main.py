import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
torch.manual_seed(0)

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):

        positive_file_path='./Positives'
        negative_file_path='./Negatives'
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
# Load pre-trained model resnet18
model = models.resnet18(pretrained=True)

# Set the parameter
for param in model.parameters():
    param.requires_grad=False

model.fc=nn.Linear(512,2)
print(model)

# Train the model
criterion = nn.CrossEntropyLoss()
train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100)
print(train_dataset[0][0].shape)

optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad],lr=0.001)

# Calculate the accuracy
n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:
        model.train() 
        optimizer.zero_grad()
        z=model(x) 
        loss=criterion(z,y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)

    correct=0
    for x_test, y_test in validation_loader:
        model.eval()
        z=model(x_test)
        _,yhat=torch.max(z.data,1)
        correct+=(yhat==y_test).sum().item()
    accuracy=correct/N_test
    accuracy_list.append(accuracy)

# Find first four misclassified samples in the validation data
count=0
index=0
for x_test, y_test in validation_loader:
    model.eval()
    z=model(x_test)
    _,yhat=torch.max(z.data,1)
    for i in range(len(y_test)):
        if yhat[i]!=y_test[i]:
            print("sample {} predicted value: {} actual value:{}".format(count,y_test[i],yhat[i]))
            count+=1
        if count>=4:
            break
        index+=1

# Plot the loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
plt.show()

plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

plt.plot(accuracy_list)
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.show()

# find the misclassified samples in the validation data
count=0
index=0
for x_test, y_test in validation_loader:
    model.eval()
    z=model(x_test)
    _,yhat=torch.max(z.data,1)
    for i in range(len(y_test)):
        if yhat[i]!=y_test[i]:
            print("sample {} predicted value: {} actual value:{}".format(index,y_test[i],yhat[i]))
            # show the sample
            plt.imshow(x_test[i][0], cmap='gray')
            plt.title("actual value: {} predicted value:{}".format(y_test[i].item(),yhat[i].item()))
            plt.show()
            count+=1
        if count>=4:
            break
        index+=1





