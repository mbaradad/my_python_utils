import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision.transforms import ToTensor


def default_loader(path):
  return Image.open(path).convert('RGB')


class CombinedDataset(data.Dataset):
  def __init__(self, datasets):
    assert type(datasets) is list
    self.datasets = datasets

    self.i_to_dataset_and_sample = dict()

    total_i = 0
    for cur_dataset in datasets:
      for k in range(len(cur_dataset)):
        self.i_to_dataset_and_sample[total_i] = (k, cur_dataset)
        total_i += 1

    self.total_samples = total_i

  def __len__(self):
    return self.total_samples

  def __getitem__(self, item):
    i, dataset = self.i_to_dataset_and_sample[item]
    return dataset[i]

class ImageFilelist(data.Dataset):
  def __init__(self, imlist, transform=ToTensor(), image_loader=default_loader, return_filename=False):
    self.transform = transform
    self.loader = image_loader
    self.imlist = imlist
    self.return_filename = return_filename

  def __getitem__(self, index):
    impath= self.imlist[index]
    img = self.loader(impath)
    if self.transform is not None:
      img = self.transform(img)
    if self.return_filename:
      return img, impath
    else:
      return img

  def __len__(self):
    return len(self.imlist)