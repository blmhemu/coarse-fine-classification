#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Import Libraries.
# Torch.
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Torchvision.
import torchvision.transforms as transforms
from torchvision import transforms, datasets, models
# Numpy, Matplotlib and other utility functions.
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Helpher Functions.
# Imshow for Tensor.
def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# A small function for training.
def train_model(model, dataloaders, dataset_sizes, 
    criterion, optimizer, scheduler, num_epochs=14):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

# Prints the model.
def print_model(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", 
            model.state_dict()[param_tensor].size())

# Prints the accuracy of a model on a loader.
def get_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test accuracy : %d %%' % (100 * correct / total))

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Prepare the data for coarse classification.
# Transforms , Can also augment the data.
data_transform_tr = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5])
    ])

trainset = datasets.ImageFolder(root='asm_data', 
    transform=data_transform_tr)
trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size=4, shuffle=True, num_workers=2)

data_transform_va = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5])
    ])

valset = datasets.ImageFolder(root='asm_data_val', 
    transform=data_transform_va)
valloader = torch.utils.data.DataLoader(valset, 
    batch_size=4, shuffle=True, num_workers=2)
testset = datasets.ImageFolder(root='asm_data_test', 
    transform=data_transform_va)
testloader = torch.utils.data.DataLoader(testset, 
    batch_size=4, shuffle=False, num_workers=2)

# Compress the above into a dict.
dataloaders = {'train':trainloader, 'val':valloader}
dataset_sizes = {'train':len(trainset), 'val':len(valset)}

# Ofcourse the labels. (Here, just the folder names)
classes = trainset.classes

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Show a Sample Batch.
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images), 
    title=[classes[labels[j]] for j in range(4)])

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Already trained ? We covered you ... We will use this to save you model.
model_file = 'models/model_34_full'
# 'n' : DO NOT TRAIN. Anything else for train.
do_train = 'n'
# Custom CNN
model_coarse = models.resnet34(pretrained=True)
num_ftrs = model_coarse.fc.in_features
model_coarse.fc = nn.Linear(num_ftrs, len(classes))
if model_file != None and os.path.isfile(model_file):
    print("File found.")
    model_coarse.load_state_dict(torch.load(model_file))
    model_coarse.eval()
    print_model(model_coarse)
    print("Model loaded.")
else:
    print("File does not exists.")
if do_train != 'n':
    print("Training started.")
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_coarse.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_coarse = train_model(model_coarse, dataloaders, dataset_sizes,
    criterion, optimizer_ft, exp_lr_scheduler, num_epochs=4)
    print("Training Done.")
    # Save the model else you are doomed
    if model_file != None:
        torch.save(model_coarse.state_dict(), model_file)
        print("Model saved to ", model_file)

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Visually see if the net has learnt something.
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] 
    for j in range(4)))
outputs = model_coarse(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%s' % classes[predicted[j]] 
    for j in range(4)))

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Get accuracy details.
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model_coarse(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            class_total[labels[i]] += 1
            class_correct[labels[i]] += c[i].item()
for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % 
        (classes[i], 100 * class_correct[i] / class_total[i]))
print('Overall Accuracy : %d %%' % 
    (100*sum(class_correct)/sum(class_total)))

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Data preparation for fine classification.
fine_trainset = {c:datasets.ImageFolder(os.path.join('asm_data', c), 
    data_transform_tr) for c in classes}
fine_valset = {c:datasets.ImageFolder(os.path.join('asm_data_val', c), 
    data_transform_va) for c in classes}
fine_testset = {c:datasets.ImageFolder(os.path.join('asm_data_test', c), 
    data_transform_va) for c in classes}
fine_trainloader = {c:torch.utils.data.DataLoader(fine_trainset[c], 
    batch_size=4, shuffle=True, num_workers=2) for c in classes}
fine_valloader = {c:torch.utils.data.DataLoader(fine_valset[c], 
    batch_size=4, shuffle=True, num_workers=2) for c in classes}
fine_testloader = {c:torch.utils.data.DataLoader(fine_testset[c], 
    batch_size=1, shuffle=False, num_workers=2) for c in classes}

fine_classes = {c:fine_trainset[c].classes for c in classes}

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Copy coarse models to use conv layers as features for fine classification.
model_fine = {c:copy.deepcopy(model_coarse) for c in classes}
for c in classes:
    model_fine[c].eval()
    for param in model_fine[c].parameters():
        param.requires_grad = False
    num_ftrs = model_fine[c].fc.in_features
    model_fine[c].fc = nn.Linear(num_ftrs, 256)
    new_layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(256, len(fine_classes[c])),
        )
    model_fine[c] = nn.Sequential(model_fine[c], new_layers)

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Train just a fc layer for species of all coarse classes.
for c in classes:
    model_file = 'models/model_34_full_' + c
    do_train = 'y'
    if model_file != None and os.path.isfile(model_file):
        print("File found.")
        model_fine[c].load_state_dict(torch.load(model_file))
        model_fine[c].eval()
        print("Model loaded.")
    else:
        print("File does not exists.")
    if do_train != 'n':
        print("Training started.")
        dataloaders = {'train':fine_trainloader[c], 'val':fine_valloader[c]}
        dataset_sizes = {'train':len(fine_trainset[c]), 'val':len(fine_valset[c])}
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_fine[c].parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model_fine[c] = train_model(model_fine[c], dataloaders, dataset_sizes,
        criterion, optimizer_ft, exp_lr_scheduler, num_epochs=4)
        print("Training Done.")
        if model_file != None:
            torch.save(model_fine[c].state_dict(), model_file)
            print("Model saved to ", model_file)

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Fine classes accuracy
for c in classes:
    print(c, end=' '),
    get_accuracy(model_fine[c], fine_testloader[c])

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Total Accuracy : Coarse + Fine
correct = 0
total = 0
for i, c in enumerate(classes, 0):
    with torch.no_grad():
        for data in fine_testloader[c]:
            images, labels  = data
            outputs_coarse = model_coarse(images)
            _, predicted_coarse = torch.max(outputs_coarse.data, 1)
            total += labels.size(0)
            if predicted_coarse == i:
                outputs_fine = model_fine[c](images)
                _, predicted_fine = torch.max(outputs_fine.data, 1)
                correct += (predicted_fine == labels).sum().item()
print('Total test accuracy : %d %%' % (100 * correct / total))

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
