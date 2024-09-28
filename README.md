# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## How To

### Jupyter Notebook

- The Notebook is accessible at [Image Classifier Project.ipynb](Image%20Classifier%20Project.ipynb)
- Download the [_flowers_ assets library](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) to be able to use multiple images for training, validating and testing.

### Scripts

#### Train

- `train.py` allows one to train a neural network using VGG13, VGG19 and stores the weights in a checkpoint

> Command Examples are:

```
python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 5 --arch "vgg13" --dropout 0.25 --gpu
```

#### Predict

- `predict.py` allows one to use the recently trained neural network and pass on an image to classify against the weights explored

> Command Examples are:

```
python predict.py flowers/train/102/image_08047.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
python predict.py flowers/train/1/image_06773.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
python predict.py flowers/train/5/image_05153.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
python predict.py flowers/train/10/image_07099.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
python predict.py flowers/train/15/image_06355.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
python predict.py flowers/train/20/image_04897.jpg checkpoint --category_names cat_to_name.json --top_k 5 --gpu
```
