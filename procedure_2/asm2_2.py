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
from PIL import Image
import time
import copy
import os
import natsort

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
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
    plt.pause(2)  # pause a bit so that plots are updated

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
data_transform_tr = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5])
    ])

data_transform_va = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5])
    ])

trainset = datasets.ImageFolder(root='asm_data_train_alt', 
    transform=data_transform_tr)
valset = datasets.ImageFolder(root='asm_data_val_alt', 
    transform=data_transform_va)
testset = datasets.ImageFolder(root='test', 
    transform=data_transform_va)

trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size=4, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, 
    batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, 
    batch_size=2, shuffle=False, num_workers=2)

# Compress the above into a dict.
dataloaders = {'train':trainloader, 'val':valloader}
dataset_sizes = {'train':len(trainset), 'val':len(valset)}

# Ofcourse the labels. (Here, just the folder names)
classes = trainset.classes

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Already trained ? We covered you ... We will use this to save you model.
model_file = 'models_working/Agg Models/model_34_agg_full'
# 'n' : DO NOT TRAIN. Anything else for train.
do_train = 'n'
# Custom CNN
model_agg = models.resnet34(pretrained=True)
num_ftrs = model_agg.fc.in_features
print(num_ftrs)
model_agg.fc = nn.Linear(num_ftrs, num_ftrs)
new_layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, len(classes)),
        )
model_agg = nn.Sequential(model_agg, new_layers)
if model_file != None and os.path.isfile(model_file):
    print("File found.")
    model_agg.load_state_dict(torch.load(model_file))
    model_agg.eval()
    print_model(model_agg)
    print("Model loaded.")
else:
    print("File does not exists.")
if do_train != 'n':
    print("Training started.")
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_agg.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_agg = train_model(model_agg, dataloaders, dataset_sizes,
    criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    print("Training Done.")
    # Save the model else you are doomed
    if model_file != None:
        torch.save(model_agg.state_dict(), model_file)
        print("Model saved to ", model_file)

#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#

#%%
# Visually see

dataiter = iter(testloader)
images, labels = dataiter.next()
print(images)
imshow(torchvision.utils.make_grid(images))
outputs = model_agg(images)
print(outputs)
_, predicted = torch.max(outputs, 1)
print(' '.join(classes[p] for p in predicted))



#///////////////////////////////////////////////////////////////#
#///////////////////////////////////////////////////////////////#


#%%
test_folder = 'testDatasetAssgn2'
output_file = 'output2.txt'
out_fd = open(output_file, 'w')
for root, dirs, files in os.walk(test_folder):
    for f in natsort.natsorted(files):
        file_t = test_folder + '/' + f
        img = Image.open(file_t)
        img = data_transform_va(img)
        img.unsqueeze_(0)
        output = model_agg(img)
        _, predicted = torch.max(output, 1)
        pred_class = classes[predicted].split('_')
        out_fd.write(f + ' ' + pred_class[0] + ' ' + pred_class[0] + '@' + pred_class[1] + '\n')
out_fd.close()
