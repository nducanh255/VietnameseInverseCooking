import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self, recipes, train, test):
    self.recipes = recipes
    self.train = train
    self.test = test

    ids = [recipe['id'] for recipe in self.recipes]
    self.all_image_names = ids

    labels = [recipe['label'] for recipe in self.recipes]
    self.all_labels = np.array(labels)

    self.train_ratio = int(0.8 * len(self.recipes))
    self.valid_ratio = len(self.recipes) - self.train_ratio

    if self.train == True:
      print(f'Number of training images: {self.train_ratio}')
      self.image_names = list(self.all_image_names[:self.train_ratio])
      self.labels = list(self.all_labels[:self.train_ratio])

      self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 400)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
      ])

    elif self.train == False and self.test == False:
      print(f'Number of valdiation images: {self.valid_ratio}')
      self.image_names = list(self.all_image_names[:-self.valid_ratio])
      self.labels = list(self.all_labels[:-self.valid_ratio])

      self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
      ])

    elif self.test == True and self.train == False:
      self.image_names = list(self.all_image_names[-10:])
      self.labels = list(self.all_labels[-10:])

      self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
      ])

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, index):
    image = cv2.imread(f'dataset/Images/{self.image_names[index]}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = self.transform(image)
    targets = self.labels[index]

    return {
      'image': torch.tensor(image, dtype=torch.float32),
      'label': torch.tensor(targets, dtype=torch.float32)
    }
