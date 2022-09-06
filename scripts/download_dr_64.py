# https://www.kaggle.com/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images
# wget -O image https://www.google.com/doodles/gerardus-mercators-503rd-birthday

#!/usr/bin/env python
# coding: utf-8

# # Dataset generator
# This will code will be used to output datasets (MNIST or CIFAR10) in " " separated formated.

# ### Setup

# In[1]:
import random
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Let us build some helpers. Function to get dataset as numpy array from a loader, a function to save data to file with space delimiter, and a function to convert labels into a vector of one hot labels.

# In[2]:

img_width = 64

def get_dataset(loader):
  images, labels = [], []
  for img, label in loader:
    images.append(img)
    labels.append(label)
  return torch.cat(images).numpy(), torch.cat(labels).numpy()


def save_to_file(tensor, filename):
    np.savetxt(fname=filename, delimiter=" ", X=tensor.flatten().tolist())
    

def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, 5))
    one_hot_labels[np.arange(labels.size),labels] = 1
    return one_hot_labels

# DR Dataset
dataset19 = pd.read_csv('../files/DR/labels/trainLabels19.csv')
print(dataset19)

# Importing Data 2015
dataset15 = pd.read_csv('../files/DR/labels/trainLabels15.csv')
dataset15.columns = ['id_code', 'diagnosis']

# Balancing Data
level_0 = dataset19[dataset19.diagnosis == 0].sample(n=900)
level_2 = dataset19[dataset19.diagnosis == 2].sample(n=900)

level_1 = dataset15[dataset15.diagnosis == 1].sample(n=530)
level_3 = dataset15[dataset15.diagnosis == 3].sample(n=707)
level_4 = dataset15[dataset15.diagnosis == 4].sample(n=605)

dataset19 = dataset19[dataset19['diagnosis'] > 0]
dataset19 = dataset19[dataset19['diagnosis'] != 2]
print(dataset19['diagnosis'].value_counts())

dataset19 = pd.concat([level_0, level_2, dataset19])
dataset19 = dataset19.sample(frac=1)
print(dataset19['diagnosis'].value_counts())

dataset15 = pd.concat([level_1, level_3, level_4])
dataset15 = dataset15.sample(frac=1)

print(dataset15['diagnosis'].value_counts())

# IMPORTING SELECTED IMAGES FROM THE DATASET
# RESIZING THE IMPORTING DATA
images = []
for i, image_id in enumerate(tqdm(dataset19.id_code)):
    im = cv2.imread(f'../files/DR/resized train 19/{image_id}.jpg')
    im = cv2.resize(im, (img_width, img_width))
    images.append(im)

for i, image_id in enumerate(tqdm(dataset15.id_code)):
    im = cv2.imread(f'../files/DR/resized train 15/{image_id}.jpg')
    im = cv2.resize(im, (img_width, img_width))
    images.append(im)

# APPLYING GAUSSIAN BLUR NOISE FILTER
# This function will act as a filter for the image data
def load_colorfilter(image, sigmaX=10):
    # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, img_width)
    return image


for i in range(len(images)):
    output = load_colorfilter(images[i])
    images[i] = output

images = np.array(images)
print('shape of images: ', images.shape)

# VISUALIZING BALANCED DATASET
dataset = pd.concat([dataset19, dataset15])
print(dataset['diagnosis'].value_counts())

# SCALING/NORMALISING IMAGE DATASET
X = images / 255.0
y = dataset.diagnosis.values

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

path = "../files/DR/"
train_images = X
train_labels = y
test_images = X_test
test_labels = y_test

# Save the images to file and convert the labels into one hot before saving. Repeat for test data.
print(train_images.shape)
print(one_hot(train_labels).shape)

save_to_file(train_images, path+"train_data")
save_to_file(one_hot(train_labels), path+"train_labels")

print(test_images.shape)
print(one_hot(test_labels).shape)

save_to_file(test_images, path+"test_data")
save_to_file(one_hot(test_labels), path+"test_labels")

