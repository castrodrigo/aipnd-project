import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import VGG13_Weights, VGG19_Weights

import argparse

# Example command
# python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 5 --arch "vgg13" --dropout 0.25 --gpu

### Script Argument Parser

DEFAULT_DATA_DIR = 'flowers'
DEFAULT_CHECKPOINT_DIR = '.'
DEFAULT_MODEL_ARCH = 'vgg19'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.25
DEFAULT_HIDDEN_UNITS = 512
DEFAULT_OUTPUT_UNITS = 102
DEFAULT_EPOCHS = 5
DEFAULT_GPU_STATE = False # off is default

parser = argparse.ArgumentParser()

# Data Dir
parser.add_argument('data_dir', 
                    action='store',
                    default = DEFAULT_DATA_DIR,
                    help = 'Root folder of training data location')

# Checkpoint
parser.add_argument('--save_dir', 
                    action = "store", 
                    dest = "save_dir", 
                    default = DEFAULT_CHECKPOINT_DIR)

# Model
parser.add_argument('--arch', 
                    action = "store", 
                    dest = "arch", 
                    default = DEFAULT_MODEL_ARCH,
                    help = '[vgg13,vgg19] are supported')

# Learning Rate
parser.add_argument('--learning_rate', 
                    action = "store", 
                    dest = "learning_rate", 
                    default = DEFAULT_LEARNING_RATE)

# Dropout
parser.add_argument('--dropout', 
                    action = "store", 
                    dest = "dropout", 
                    default = DEFAULT_DROPOUT)

# Hidden Units
parser.add_argument('--hidden_units', 
                    action = "store", 
                    dest = "hidden_units", 
                    default = DEFAULT_HIDDEN_UNITS)

# Output Units
parser.add_argument('--output_units', 
                    action = "store", 
                    dest = "output_units", 
                    default = DEFAULT_OUTPUT_UNITS)


# Epochs
parser.add_argument('--epochs', 
                    action = "store", 
                    dest = "epochs", 
                    default = DEFAULT_EPOCHS)

# GPU
parser.add_argument('--gpu', 
                    action = "store_true", 
                    dest = "gpu", 
                    default = DEFAULT_GPU_STATE)


user_input = parser.parse_args()


### Setup 

## Device
# Validate device: MacBook x Nvidia x noGpu
if user_input.gpu:
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
  device = torch.device("cpu")
  
## Folders
train_dir = user_input.data_dir + '/train'
valid_dir = user_input.data_dir + '/valid'
test_dir = user_input.data_dir + '/test'

## Training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform = data_transforms['test']),
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64),
}

## Model Setup

# Architecture
if user_input.arch.lower() == 'vgg13':
  model = models.vgg13(weights = VGG13_Weights.DEFAULT)
else: 
  model = models.vgg19(weights = VGG19_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

# Override Classifier
model.classifier = nn.Sequential(nn.Linear(25088, int(user_input.hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(float(user_input.dropout)),
                                 nn.Linear(int(user_input.hidden_units), int(user_input.output_units)),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Define Adam optimizer for Sequential Classifier overriden
optimizer = optim.Adam(model.classifier.parameters(), lr = float(user_input.learning_rate))

# Force model to run on identified device
model.to(device)

def validate_model_with_dataloader(model, dataloader, criterion, device):
    
    ''' Returns training loss and acuracy for a given run of the model in a give subset of data
    '''
    
    dataloader_output_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            
             # Force images and labels to identified device
            images, labels = images.to(device), labels.to(device)

            dataloader_output = model.forward(images)
            batch_loss = criterion(dataloader_output, labels)

            dataloader_output_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(dataloader_output)
            equals = (labels.data == ps.max(dim = 1)[1])
            accuracy += equals.type(torch.FloatTensor).mean()
    
    return dataloader_output_loss, accuracy

## Train Model

epochs = int(user_input.epochs)
steps = 0
running_loss = 0
print_every = 10

for epoch in range(epochs):
    for images, labels in dataloaders['train']:
        steps += 1
        
        # Force images and labels to identified device
        images, labels = images.to(device), labels.to(device)
        
        # Fowards Operation
        run_output = model.forward(images)
        loss = criterion(run_output, labels)
        
        # Backwards Operation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            # Disable training
            model.eval()
            
            with torch.no_grad():
                    
                test_loss, accuracy = validate_model_with_dataloader(
                    model, dataloaders['valid'], criterion, device)
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                  f"Test accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            
            running_loss = 0
            
            # Enable training
            model.train()
            
## Validate Trained Model

with torch.no_grad():

    _, run_accuracy = validate_model_with_dataloader(
                    model, dataloaders['test'], criterion, device)

print(f"\n\nModel training accuracy with test dataset is: {100 * run_accuracy/len(dataloaders['test']):.2f}%")

## Checkpoint Save

model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {
              'arch': user_input.arch,
              'input_size': 25088,
              'n_size': int(user_input.hidden_units),
              'output_size': int(user_input.output_units),
              'dropout': float(user_input.dropout),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(
    checkpoint
    , user_input.save_dir + '/checkpoint.pth'
)
