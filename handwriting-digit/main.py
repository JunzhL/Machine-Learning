import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Plot the model parameters for each class
def PlotParameters(model): 
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            ax.set_xlabel("class: {0}".format(i))
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()

# Plot the data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

# training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# print("Print the training dataset:\n ", train_dataset)

# validation dataset
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
# print("Print the validation dataset:\n ", validation_dataset)

# Print the first image and label
# print("First Image and Label") 
# show_data(train_dataset[0])

# Print the label
# print("The label: ", train_dataset[3][1])

# The third sample
# show_data(train_dataset[2])

class SoftMax(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        z = self.linear(x)
        return z
# print(train_dataset[0][0].shape)

# input size and output size
input_dim = 28 * 28
output_dim = 10

# Create the model
model = SoftMax(input_dim, output_dim)
# print("Print the model:\n ", model)

# parameters
# print('W: ',list(model.parameters())[0].size())
# print('b: ',list(model.parameters())[1].size())
# PlotParameters(model)

# Make a prediction
# get the X value of the first image
X = train_dataset[0][0]
# print(X.shape)
X = X.view(-1, 28*28)
# print(X.shape)

# Define the learning rate, optimizer, criterion, and data loader
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Train the model, output the value assigned to each class
model_output = model(X)
actual = torch.tensor([train_dataset[0][1]])
show_data(train_dataset[0])
print("Output: ", model_output)
print("Actual:", actual)
print("Criterion: ", criterion(model_output, actual))

# model_output are not probabilities
softmax = nn.Softmax(dim=1)
probability = softmax(model_output)
print(probability)

# Negative log
print("Negative Log: ", -1*torch.log(probability[0][actual]))

n_epochs = 10
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)

def train_model(n_epochs):
    # Loops n_epochs times
    for epoch in range(n_epochs):
        # For each batch in loader
        for x, y in train_loader:
            # Reset the calculated gradient value
            optimizer.zero_grad()
            # Makes a prediction based on the image tensor
            z = model(x.view(-1, 28 * 28))
            # Calculates loss
            loss = criterion(z, y)
            # Calculate gradient value with each weight and bias
            loss.backward()
            # Updates the weight and bias
            optimizer.step()
        
        # check accuracy
        correct = 0
        for x_test, y_test in validation_loader:
            # Makes prediction
            z = model(x_test.view(-1, 28 * 28))
            # Finds the class with the higest output
            _, pred = torch.max(z.data, 1)
            correct += (pred == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)

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

PlotParameters(model)

# misclassified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, pred = torch.max(z, 1)
    if pred != y:
        show_data((x, y))
        plt.show()
        print("pred:", pred)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break       

# classified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, pred = torch.max(z, 1)
    if pred == y:
        show_data((x, y))
        plt.show()
        print("pred:", pred)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break  
