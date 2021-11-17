import torch.utils.data as data

from PIL import Image
import os
import os.path


def default_loader(path):
  return Image.open(path).convert('RGB')

from torchvision.transforms import ToTensor

class ImageFilelist(data.Dataset):
  def __init__(self, imlist, transform=ToTensor(), loader=default_loader):
    self.transform = transform
    self.loader = loader
    self.imlist = imlist

  def __getitem__(self, index):
    impath= self.imlist[index]
    img = self.loader(impath)
    if self.transform is not None:
      img = self.transform(img)

    return img

  def __len__(self):
    return len(self.imlist)