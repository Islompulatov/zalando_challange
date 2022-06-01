from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
# import data
# from test_model import model_class
import data_handler as dh
from model_train import model_check
torch.manual_seed(42)

# 1. DATA
trainloader, testloader = dh.load_data("~/.pytorch/F_MNIST_data/")
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
# # Download and load the training data
# trainset = datasets.FashionMNIST(
#     '~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=64, shuffle=True)

# # Download and load the test data
# testset = datasets.FashionMNIST(
#     '~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

model = model_check(784, 400, 200, 10).cuda()
# TRAINING AND VALIDATION
learning_rate = 0.001
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# image, label = next(iter(trainloader))
train_losses = []
test_losses = []
accuracies = []
for epoch in tqdm(range(epochs)):

    running_loss = 0
    # training
    for x_train_batch, y_train_batch in trainloader:

        optimizer.zero_grad()
        # forward pass
        logits = model(x_train_batch.view(x_train_batch.shape[0], -1))

        # loss
        train_loss = criterion(logits, y_train_batch)
        running_loss += train_loss.item()

        # backward pass
        train_loss.backward()

        optimizer.step()

    # mean loss (all batches losses divided by the total number of batches)
    train_losses.append(running_loss/len(trainloader))

    # validation
    model.eval()
    with torch.no_grad():
        running_accuracy = 0
        running_loss = 0

        for x_test_batch, y_test_batch in testloader:

            # logits
            test_logits = model(
                x_test_batch.view(x_test_batch.shape[0], -1))

            # predictions
            test_preds = torch.argmax(test_logits, dim=1)

            # running accuracy
            running_accuracy += accuracy_score(y_test_batch, test_preds)

            # loss
            test_loss = criterion(test_logits, y_test_batch)
            running_loss += test_loss.item()

        # mean accuracy for each epoch
        accuracies.append(running_accuracy/len(testloader))

        # mean loss for each epoch
        test_losses.append(running_loss/len(testloader))

    model.train()


# Plots
x_epochs = list(range(epochs))
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(x_epochs, train_losses, marker='o', label='train')
plt.plot(x_epochs, test_losses, marker='o', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, accuracies, marker='o',
         c='red', label='test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
