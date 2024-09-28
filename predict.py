import matplotlib.pyplot as plt

from PIL import Image

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import VGG13_Weights, VGG19_Weights

import argparse

import json

# Example commands
# python predict.py flowers/train/102/image_08047.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/train/1/image_06773.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/train/5/image_05153.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/train/10/image_07099.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/train/15/image_06355.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
# python predict.py flowers/train/20/image_04897.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu

### Script Argument Parser

DEFAULT_INPUT_PATH = 'flowers/train/102/image_08047.jpg'
DEFAULT_CHECKPOINT_LOCATION = 'checkpoint'
DEFAULT_CATEGORY_MAPPER_FILE = 'cat_to_name.json'
DEFAULT_TOP_K = 5
DEFAULT_GPU_STATE = False # off
 
parser = argparse.ArgumentParser()

# Data Dir
parser.add_argument('image_path', 
                    action='store',
                    default = DEFAULT_INPUT_PATH,
                    help = 'Path to image')

# Checkpoint
parser.add_argument('checkpoint', 
                    action = "store", 
                    default = DEFAULT_CHECKPOINT_LOCATION,
                    help = 'Path to checkpoint file with its name')

# TOP K n
parser.add_argument('--top_k', 
                     dest="top_k", 
                     default = DEFAULT_TOP_K)

# Category names file location
parser.add_argument('--category_names', 
                     action = "store", 
                     dest = "category_names", 
                     default = DEFAULT_CATEGORY_MAPPER_FILE)

# GPU
parser.add_argument('--gpu', 
                    action = "store_true", 
                    dest = "gpu", 
                    default = DEFAULT_GPU_STATE)


user_input = parser.parse_args()

## Device

# Validate device: MacBook x Nvidia x noGpu
if user_input.gpu:
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
  device = torch.device("cpu")

## Categories

with open(user_input.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
## Load Model Checkpoint

checkpoint = torch.load(user_input.checkpoint + '.pth', weights_only = False)

if checkpoint['arch'] == 'vgg13':
  model = models.vgg13(weights = VGG13_Weights.DEFAULT)
else: 
  model = models.vgg19(weights = VGG19_Weights.DEFAULT)

model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['n_size']),
                                 nn.ReLU(),
                                 nn.Dropout(checkpoint['dropout']),
                                 nn.Linear(checkpoint['n_size'], checkpoint['output_size']),
                                 nn.LogSoftmax(dim=1))

model.class_to_idx = checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])

## Image functions

DIM_DEFAULT = 256
DIM_CROP_DEFAULT = 224
MEAN_DEFAULT = [0.485, 0.456, 0.406]
STD_DEFAULT = [0.229, 0.224, 0.225]

def preprocess_image_for_model(image_input):
  ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
      returns an Numpy array
  '''
      
  image = Image.open(image_input)

  # Get dimensions
  width, height = image.size
  
  # Resize based on smaller side
  if width > height:
      image.resize((width*DIM_DEFAULT//height, DIM_DEFAULT), Image.Resampling.LANCZOS)
  else:
      image.resize((DIM_DEFAULT, height*DIM_DEFAULT//width), Image.Resampling.LANCZOS)
  
  left = (width - DIM_CROP_DEFAULT)/2
  top = (height - DIM_CROP_DEFAULT)/2
  right = (width + DIM_CROP_DEFAULT)/2
  bottom = (height + DIM_CROP_DEFAULT)/2

  # Center image and crop it
  image = image.crop((left, top, right, bottom))
  
  np_image = np.array(image)
  
  # Channel normalisation
  np_image = np_image/255
  
  # Input required mean and deviation
  mean = np.array(MEAN_DEFAULT)
  std = np.array(STD_DEFAULT)
  
  # Transpose image
  np_image = ((np_image - mean)/std).transpose((2, 0, 1))
  
  return np_image

def imshow(image, ax=None, title=None):
  """Imshow for Tensor."""
  if ax is None:
      fig, ax = plt.subplots()
  
  # PyTorch tensors assume the color channel is the first dimension
  # but matplotlib assumes is the third dimension
  image = image.transpose((1, 2, 0))
  
  # Undo preprocessing
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = std * image + mean
  
  # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
  image = np.clip(image, 0, 1)
  
  ax.imshow(image)
  
  return ax

def get_model_predictions_for_image(image_path, model, topk = 5):
  ''' Predict the class (or classes) of an image using a trained deep learning model.
  '''
  with torch.no_grad():
    model.eval()
    
    image = preprocess_image_for_model(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    run_output = model.forward(image)
    
    ps = torch.exp(run_output)
    top_p, top_class = ps.topk(topk)
  
  return top_p, top_class

def validate_image_with_model(image_location, model, topk):
    
  probs, classes = get_model_predictions_for_image(image_location, model, int(topk))
  
  probs = probs.detach().numpy()[0]
  classes = classes.detach().numpy()[0]
  
  imshow(preprocess_image_for_model(image_location), plt.subplot(2,1,1))

  flowers = [i for i in classes]
  categories_flowers = []
  for flower in flowers:
      for key, value in checkpoint['class_to_idx'].items():
          if value == flower:
              categories_flowers.append(key)

  flower_names = [cat_to_name[i] for i in categories_flowers]

  axis_graph = plt.subplot(2,1,2)
  axis_graph.barh(flower_names, probs)
  plt.tight_layout(pad = 0.8)

  plt.show()

## Render Input Classification
validate_image_with_model(user_input.image_path, model, user_input.top_k)