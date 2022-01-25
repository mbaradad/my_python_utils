import torch.utils.data as data

from PIL import Image
import os
import os.path


def default_loader(path):
  return Image.open(path).convert('RGB')

from torchvision.transforms import ToTensor

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