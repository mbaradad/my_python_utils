#force 3.5 div to make it compatible with both 2.7 and 3.5
from __future__ import division
# we need to import this before torch:
# https://github.com/pytorch/pytorch/issues/19739
try:
  from nbconvert.exporters import pdf
  import open3d as o3d
  from plyfile import PlyData, PlyElement
except:
  # print("Failed to import some optional packages from my_python_utils, some plotting/visualization functions may fail!")
  pass

import cv2
import random

def randint_replacement(*args, **kwargs): raise Exception("Don't use random.randint as it can sample high, use numpy.random.randint")
random.randint = randint_replacement

import torch
from pathlib import Path
import git

try:
  import cPickle as pickle
except:
  import _pickle as pickle
import os

try:
  import seaborn as sns
except Exception as e:
  print("Failed to import seaborn. Probably requirements_extra from common_utils were not installed!")

import glob
import shutil
import time
import math
import sys

import warnings
import random
import argparse
try:
  import matplotlib
except:
  pass
from tqdm import tqdm

from scipy import misc as scipy_misc
import struct
from pathlib import Path

from scipy.ndimage.filters import gaussian_filter
from multiprocessing.pool import ThreadPool

import imageio
import re

import numpy as np
from PIL import Image, ImageDraw

import datetime
import json
import difflib

import torch.nn.functional as F
from torch.autograd import Variable

from scipy.linalg import lstsq
import socket

# other utils
from my_python_utils.vis_utils.visdom_visualizations import *
from my_python_utils.logging_utils import *

from sklearn.manifold import TSNE

from multiprocessing.pool import ThreadPool

import GPUtil
import tempfile

from p_tqdm import p_map
import hashlib

import contextlib
from my_python_utils.geom_utils import *

import subprocess

# to be able to do np.savez_compressed(filename, dict_of(a,b,c))
from sorcery import dict_of

global VISDOM_BIGGEST_DIM
VISDOM_BIGGEST_DIM = 600

# dictionary that can be accessed with .attribute_name as well as ['attribute_name']
class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

# to do with no_context:, to keep syntax but remove the effect while debugging
# for example with torch.no_grad() -> with no_context():
no_context = contextlib.suppress
empty_context = contextlib.suppress

def get_hostname():
  return socket.gethostname()

def get_conda_env():
  assert 'anaconda' in sys.executable and sys.executable
  return sys.executable.split('/')[-3]

def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  gpus_arg = str(gpus_arg)
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

  flag = 0
  for i in range(len(gpus)):
    for i1 in range(len(gpus)):
      if i != i1:
        if gpus[i] == gpus[i1]:
          flag = 1
  assert not flag, "Gpus repeated: {}".format(gpus)

  return gpus

def gettimedatestring():
  return datetime.datetime.now().strftime("%m-%d-%H:%M")

def get_time_microseconds():
  from datetime import datetime
  dt = datetime.now()
  return dt.microsecond


class ThreadedMultiqueueSafer():
  def __init__(self, safe_func, use_threading=True, queue_size=20, n_workers=20):
    self.queues = [Queue(queue_size) for _ in range(n_workers)]
    self.safe_func = safe_func
    self.use_threading = use_threading
    self.n_workers = n_workers
    self.processes = []

    def safe_results_process(queue, safe_func):
      while True:
        if queue.empty():
          time.sleep(0.1)
        else:
          try:
            actual_safe_dict = queue.get()
            safe_func(**actual_safe_dict)
            continue
          except Exception as e:
            print(e)

    if self.use_threading:
      for i in range(n_workers):
        p = Process(target=safe_results_process, args=[self.queues[i], self.safe_func])
        self.processes.append(p)
        p.start()

  def put_safe_dict(self, safe_dict):
    if self.use_threading:
      i = np.random.randint(0, self.n_workers)
      while self.queues[i].full():
        print("Safer {} queue is full, waiting...".format(i))
        time.sleep(1)
        i = np.random.randint(0, self.n_workers)
      self.queues[i].put(safe_dict)
    else:
      self.safe_func(**safe_dict)

  def queues_empty(self):
    return all([queue.empty() for queue in self.queues])

  def destroy(self):
    if not self.queues_empty():
      print("Not all queues are empty, won't destroy!")
      return
    else:
      for p in self.processes:
        p.terminate()

  def wait_to_complete_and_destro(self, sleep_time_seconds=1):
    while not self.queues_empty():
      time.sleep(sleep_time_seconds)

    self.destroy()

def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

from multiprocessing import Lock

global lock
lock = Lock()
def thread_safe_read_text_file_lines(filename):
  global lock
  lock.acquire()
  try:
    lines = list()
    with open(filename, 'r') as f:
      for line in f:
        lines.append(line.replace('\n',''))
  finally:
    lock.release()
  return lines

def touch(file):
  Path(file).touch()

def read_text_file_lines(filename, stop_at=-1):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      if stop_at > 0 and len(lines) >= stop_at:
        return lines
      lines.append(line.replace('\n',''))
  return lines

def write_text_file_lines(lines, file):
  assert type(lines) is list, "Lines should be a list of strings"
  with open(file, 'w') as file_handler:
    for item in lines:
      file_handler.write("%s\n" % item)

def write_text_file(text, filename):
  with open(filename, "w") as file:
    file.write(text)

def read_text_file(filename):
  text_file = open(filename, "r")
  data = text_file.read()
  text_file.close()
  return data

def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5
    return array

def is_headed_execution():
  #if there is a display, we are running locally
  return 'DISPLAY' in os.environ.keys()
try:
  if not is_headed_execution():
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      matplotlib.use('Agg')
except:
  pass 

def chunk_list(seq, n_chunks):
  avg = len(seq) / float(n_chunks)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def chunk_list_max_len(seq, max_len):
  out = []
  last = 0

  while last < len(seq):
    out.append(seq[int(last):int(last + max_len)])
    last += max_len

  return out

def tsne(X, components=2):
  assert len(X.shape) == 2, "N batch X M dimensions"
  tsne = TSNE(n_components=components)
  X_embedded = tsne.fit_transform(X)
  return X_embedded

def png_16_bits_imread(file):
  return cv2.imread(file, -cv2.IMREAD_ANYDEPTH)

def cv2_imread(file, return_BGR=False, read_alpha=False):
  im = None
  if read_alpha:
    try:
      im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    except:
      print("Failed to read alpha channel, will us standard imread!")
  if not read_alpha or im is None:
      im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2,0,1)
  if return_BGR:
    return im
  if im.shape[0] == 4:
    return np.concatenate((im[:3][::-1], im[3:4]))
  else:
    return im[::-1, :, :]

def im_to_bw(image):
  assert len(image.shape) == 3
  image = (image - image.min()) / (image.max() - image.min())
  image = np.array(image * 255, dtype='uint8')
  gray = cv2.cvtColor(image.transpose((1,2,0)), cv2.COLOR_BGR2GRAY)

  return gray


def load_image_tile(filename, top, bottom, left, right, dtype='uint8'):
  #img = pyvips.Image.new_from_file(filename, access='sequential')
  roi = img.crop(left, top, right - left, bottom - top)
  mem_img = roi.write_to_memory()

  # Make a numpy array from that buffer object
  nparr = np.ndarray(buffer=mem_img, dtype=dtype,
                     shape=[roi.height, roi.width, roi.bands])
  return nparr

def cv2_imwrite(im, file, normalize=False, jpg_quality=None):
  if len(im.shape) == 3 and im.shape[0] == 3 or im.shape[0] == 4:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min())/(im.max() - im.min())
    im = np.array(255.0*im, dtype='uint8')
  if jpg_quality is None:
    # The default jpg quality seems to be 95
    if im.shape[-1] == 3:
      cv2.imwrite(file, im[:,:,::-1])
    else:
      raise Exception('Alpha not working correctly')
      im_reversed = np.concatenate((im[:,:,3:0:-1], im[:,:,-2:-1]), axis=2)
      cv2.imwrite(file, im_reversed)
  else:
    cv2.imwrite(file, im[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


def merge_side_by_side(im1, im2):
  assert im1.shape == im2.shape
  im_canvas = np.concatenate((np.zeros_like(im1), np.zeros_like(im1)), axis=2)
  im_canvas[:,:,:im1.shape[-1]] = im1
  im_canvas[:,:,im1.shape[-1]:] = im2
  return im_canvas

def is_pycharm_run():
  return'PYCHARM_RUN' in os.environ.keys()

def visdom_histogram(array, win=None, title=None, env=None, vis=None):
  if env is None:
    env = PYCHARM_VISDOM
  if type(array) is list:
    array = np.array(array)
  if vis is None:
    vis = global_vis
  array = array.flatten()

  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opt = dict()
  opt['title'] = title
  vis.histogram(array, env=env, win=win, opts=opt)


def visdom_barplot(array, env=None, win=None, title=None, vis=None):
  if env is None:
    env = PYCHARM_VISDOM
  if vis is None:
    vis = global_vis

  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opt = dict()
  opt['title'] = str(win)
  vis.bar(array, env=env, win=win, opts=opt)

def visdom_bar_plot(array, rownames=None, env=None, win=None, title=None, vis=None):
  if env is None:
    env = PYCHARM_VISDOM
  if vis is None:
    vis = global_vis

  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opt = dict()
  opt['title'] = title
  if not rownames is None:
    opt['rownames'] = rownames
  vis.bar(array, env=env, win=win, opts=opt)


def visdom_boxplot(array, env=None, win='test', title=None, vis=None):
  if env is None:
    env = PYCHARM_VISDOM
  if vis is None:
    vis = global_vis

  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opt = dict()
  vis.boxplot(array, env=env, win=win, opts=opt)

# If multiple Ys lines are provided, first dimension is # of lines, second is # of data points
def visdom_line(Ys, X=None, names=None, env=None, win=None, title=None, vis=None):
  if env is None:
    env = PYCHARM_VISDOM
  if vis is None:
    vis = global_vis

  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opt = dict()
  opt['fillarea'] = False
  if not names is None:
    opt['legend'] = names
  if not title is None:
    opt['title'] = title
  if type(Ys) is list:
    Ys = np.array(Ys)
    Ys = tonumpy(Ys).transpose()
  else:
    # inner dimension is the data, but visdom expects the opposite
    Ys = tonumpy(Ys).transpose()
  if type(X) is list:
    Xs = np.array(X)
  if len(Ys.shape) == 2:
    if Ys.shape[1] == 1:
      Ys = Ys[:,0]
    elif not X is None:
      assert len(X.shape) == 1, "X should only have one dimension, as it is common for all Ys!"
      assert X.shape[0] == Ys.shape[0], "Data and X should have the same number of points"
  Ys = np.array(Ys)
  if X is None:
    X = np.arange(Ys.shape[0])
  vis.line(Ys, X=X, env=env, win=win, opts=opt)

def save_visdom_plot(win, save_path):
  return

def touch(fname, times=None, exists_ok=True):
  if exists_ok and os.path.exists(fname):
    return
  with open(fname, 'a'):
    os.utime(fname, times)

def get_image_ressolution_fast_jpg(filename):
  return get_image_width_height_fast_jpg(filename)

def get_image_width_height_fast_jpg(filename):
  """"This function prints the resolution of the jpeg image file passed into it"""

  # open image for reading in binary mode
  with open(filename, 'rb') as img_file:
    # height of image (in 2 bytes) is at 164th position
    img_file.seek(163)

    # read the 2 bytes
    a = img_file.read(2)

    # calculate height
    height = (a[0] << 8) + a[1]

    # next 2 bytes is width
    a = img_file.read(2)

    # calculate width
    width = (a[0] << 8) + a[1]
  return width, height


class UnknownImageFormat(Exception):
  pass

def get_image_size_fast_png(file_path):
  """
  Return (width, height) for a given img file content - no external
  dependencies except the os and struct modules from core
  """
  size = os.path.getsize(file_path)

  with open(file_path) as input:
    height = -1
    width = -1
    data = input.read(25)

    if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
      # GIFs
      w, h = struct.unpack("<HH", data[6:10])
      width = int(w)
      height = int(h)
    elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
          and (data[12:16] == 'IHDR')):
      # PNGs
      w, h = struct.unpack(">LL", data[16:24])
      width = int(w)
      height = int(h)
    elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
      # older PNGs?
      w, h = struct.unpack(">LL", data[8:16])
      width = int(w)
      height = int(h)
    elif (size >= 2) and data.startswith('\377\330'):
      # JPEG
      msg = " raised while trying to decode as JPEG."
      input.seek(0)
      input.read(2)
      b = input.read(1)
      try:
        while (b and ord(b) != 0xDA):
          while (ord(b) != 0xFF): b = input.read(1)
          while (ord(b) == 0xFF): b = input.read(1)
          if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
            input.read(3)
            h, w = struct.unpack(">HH", input.read(4))
            break
          else:
            input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
          b = input.read(1)
        width = int(w)
        height = int(h)
      except struct.error:
        raise UnknownImageFormat("StructError" + msg)
      except ValueError:
        raise UnknownImageFormat("ValueError" + msg)
      except Exception as e:
        raise UnknownImageFormat(e.__class__.__name__ + msg)
    else:
      raise UnknownImageFormat(
        "Sorry, don't know how to get information from this file."
      )

  return width, height

def tile_images(imgs, tiles_x_y=None, tile_size_x_y=None, border_pixels=0, border_color=(0, 0, 0), border_pixels_x_y=None):
  if tile_size_x_y is None:
    tile_size_y, tile_size_x = imgs[0].shape[-2:]
  else:
    tile_size_x, tile_size_y = tile_size_x_y
  if tiles_x_y is None:
    n_tiles_x = int(np.ceil(np.sqrt(len(imgs))))
    # just so that they fit, no need to have a square grid
    n_tiles_y = int(np.ceil(len(imgs) / n_tiles_x))
    assert n_tiles_x * n_tiles_y >= len(imgs)
  else:
    n_tiles_x, n_tiles_y = tiles_x_y
  if border_pixels_x_y is None:
    border_pixels_x = border_pixels_y = border_pixels
  else:
    assert len(border_pixels_x_y) == 2, "border_pixels_x_y should have length 2"
    border_pixels_x, border_pixels_y = border_pixels_x_y

  del border_pixels

  final_img = np.zeros((3, tile_size_y * n_tiles_y + border_pixels_y * (n_tiles_y - 1), tile_size_x * n_tiles_x + border_pixels_x * (n_tiles_x - 1)))
  final_img += np.array(border_color)[:, None, None]
  n_imgs = len(imgs)
  k = 0
  for i in range(n_tiles_y):
    for j in range(n_tiles_x):
      tile = myimresize(tonumpy(imgs[k]), (tile_size_y, tile_size_x))
      if len(tile.shape) == 2:
        tile = tile[None,:,:]
      if tile.shape[0] == 1:
        tile = np.concatenate((tile, tile, tile))
      final_img[:, i*(tile_size_y + border_pixels_y):i*(tile_size_y + border_pixels_y) + tile_size_y,
                   j*(tile_size_x + border_pixels_x):j*(tile_size_x + border_pixels_x) + tile_size_x] = tile
      k = k + 1
      if k >= n_imgs:
        break
    if k >= n_imgs:
      break
  return final_img

def tile_images_pdf(imgs, pdf_output_file, tiles, tile_size, border_pixels=0, return_as_img=True):
  from fpdf import FPDF
  tile_size_x, tile_size_y = tile_size
  n_tiles_x, n_tiles_y = tiles

  w = n_tiles_x * tile_size_x + (n_tiles_x - 1) * border_pixels
  h = n_tiles_y * tile_size_y + (n_tiles_y - 1) * border_pixels
  pdf = FPDF('P', 'pt', (w, h))
  pdf.add_page()

  n_imgs = len(imgs)
  k = 0
  for i in range(n_tiles_y):
    for j in range(n_tiles_x):
      tile = myimresize(tonumpy(imgs[k]), tile_size[::-1])
      if len(tile.shape) == 2:
        tile = tile[None,:,:]
      if tile.shape[0] == 1:
        tile = np.concatenate((tile, tile, tile))
      tmp_filename = '/tmp/{}.jpg'.format(get_random_number_from_timestamp())
      cv2_imwrite(tile, tmp_filename)
      x = j * (tile_size_x + border_pixels)
      y = i * (tile_size_y + border_pixels)
      pdf.image_path(tmp_filename, x, y, tile_size_x, tile_size_y)
      k = k + 1
      if k >= n_imgs:
        break
    if k >= n_imgs:
      break
  pdf.output(pdf_output_file, "F")

  from pdf2image import convert_from_path

  images = convert_from_path(pdf_output_file)
  # to numpy array
  image = np.array(images[0]).transpose((2,0,1))


  return image

def str2intlist(v):
 if len(v) == 0:
   return []
 return [int(k) for k in v.split(',')]


def get_subdirs(directory, max_depth):
  if max_depth == 0:
    if os.path.isdir(directory):
      return [directory]
    else:
      return []
  else:
    directories = listdir(directory, prepend_folder=True, type='folder')
    all_subdirs = []
    for dir in directories:
      all_subdirs.extend(get_subdirs(dir, max_depth=max_depth - 1))
    return all_subdirs

def cross_product_mat_sub_x_torch(v):
  assert len(v.shape) == 3 and v.shape[1:] == (3,1)
  batch_size = v.shape[0]
  zero = torch.zeros((batch_size, 1))
  sub_x_mat = torch.cat((torch.cat((zero, -v[:, 2], v[:,1]), axis=1)[:,None,:],
                         torch.cat((v[:, 2], zero, -v[:,0]), axis=1)[:,None,:],
                         torch.cat((-v[:, 1],  v[:,0], zero), axis=1)[:,None,:]), axis=1)
  return sub_x_mat

  return np.array([(0, -float(v[2]), float(v[1])),
                   (float(v[2]), 0, -float(v[0])),
                   (-float(v[1]), float(v[0]), 0)])

def cross_product_mat_sub_x_np(v):
  assert v.shape == (3,1)
  return np.array([(0, -float(v[2]), float(v[1])),
                   (float(v[2]), 0, -float(v[0])),
                   (-float(v[1]), float(v[0]), 0)])

def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')

def comma_separated_str_list(v):
  if not type(v) is str:
    raise argparse.ArgumentTypeError('Should be a string.')
  else:
    v = v.replace(' ', '')
    return v.split(',')

def add_line(im, origin_x_y, end_x_y, color=(255, 0, 0), width=0):
  im = Image.fromarray(im.transpose())
  draw = ImageDraw.Draw(im)

  draw.line((origin_x_y[1], origin_x_y[0], end_x_y[1], end_x_y[0]), fill=color, width=width)

  return np.array(im).transpose()

def add_bbox(im, x_0_y_0_s, x_1_y_1_s, color=(255, 0, 0), line_width=1):
  im_with_box = np.ascontiguousarray(np.array(im).transpose((1, 2, 0)))
  if not type(x_0_y_0_s) is list:
    x_0_y_0_s = [x_0_y_0_s]
  if not type(x_1_y_1_s) is list:
    x_1_y_1_s = [x_1_y_1_s]
  for i in range(len(x_0_y_0_s)):
    x_0_y_0 = x_0_y_0_s[i]
    x_1_y_1 = x_1_y_1_s[i]
    if type(color) is list:
      actual_color = tuple(color[i])
    else:
      actual_color = color
    im_with_box = cv2.rectangle(im_with_box, tuple(np.array(x_0_y_0, dtype='int')), tuple(np.array(x_1_y_1, dtype='int')), actual_color, line_width)
  if type(im_with_box) is np.ndarray:
    return im_with_box.transpose((2, 0, 1))
  else:
    return np.array(im_with_box.get()).transpose((2, 0, 1))

def add_squared_bbox(im, centers_x_y, box_width=5, color=(255, 0, 0), line_width=1):
  im_with_box = np.array(im).transpose((1, 2, 0))
  if not type(centers_x_y) is list:
    centers_x_y = [centers_x_y]
  for i in range(len(centers_x_y)):
    center_x_y = centers_x_y[i]
    if type(color) is list:
      actual_color = tuple(color[i])
    else:
      actual_color = color
    im_with_box = cv2.rectangle(im_with_box, tuple(np.array(center_x_y, dtype='int') - box_width // 2), tuple(np.array(center_x_y, dtype='int') + box_width // 2), actual_color, line_width)
  if type(im_with_box) is np.ndarray:
    return im_with_box.transpose((2, 0, 1))
  else:
    return np.array(im_with_box.get()).transpose((2, 0, 1))

def add_text(im, lines, starts_x_y, color=(255, 0, 0), font_scale=1, line_width=2):
  im_with_text = np.ascontiguousarray(np.array(im).transpose((1, 2, 0)))
  if not type(starts_x_y) is list:
    starts_x_y = [starts_x_y]
  if not type(lines) is list:
    lines = [lines]
  assert len(lines) == len(starts_x_y)
  for i in range(len(starts_x_y)):
    start_x_y = starts_x_y[i]
    text = lines[i]
    if type(color) is list:
      actual_color = tuple(color[i])
    else:
      actual_color = color
    im_with_text = cv2.putText(im_with_text, text, tuple(np.array(start_x_y, dtype='int')), cv2.FONT_HERSHEY_SIMPLEX, font_scale, actual_color, line_width)
  if type(im_with_text) is np.ndarray:
    return im_with_text.transpose((2, 0, 1))
  else:
    return np.array(im_with_text.get()).transpose((2, 0, 1))


def add_circle(im, centers_x_y, radius=5, color=(255, 0, 0)):
  im_with_circle = np.array(im).transpose((1, 2, 0))
  if not type(centers_x_y) is list:
    centers_x_y = [centers_x_y]
  for i in range(len(centers_x_y)):
    center_x_y = centers_x_y[i]
    if type(color) is list:
      actual_color = tuple(color[i])
    else:
      actual_color = color
    im_with_circle = cv2.circle(cv2.UMat(im_with_circle), tuple(np.array(center_x_y, dtype='int')), radius, actual_color, -1).get()
  if type(im_with_circle) is np.ndarray:
    return im_with_circle.transpose((2, 0, 1))
  else:
    return np.array(im_with_circle.get()).transpose((2, 0, 1))

def add_arrow(im, origins_x_y, ends_x_y, colors=(255, 0, 0), width=5):
  assert len(im.shape) == 3 and im.shape[0] == 3, "Only implemented for colored images"

  im_with_arrow = np.array(im).transpose((1, 2, 0))

  if type(origins_x_y) is np.ndarray and len(origins_x_y.shape) == 2:
    origins_x_y = [k for k in origins_x_y]

  if type(ends_x_y) is np.ndarray and len(ends_x_y.shape) == 2:
    ends_x_y = [k for k in ends_x_y]

  if not type(origins_x_y) is list:
    origins_x_y = [origins_x_y]
  if not type(ends_x_y) is list:
    ends_x_y = [ends_x_y]

  assert len(origins_x_y) == len(ends_x_y)

  for i in range(len(origins_x_y)):
    origin_x_y = origins_x_y[i]
    end_x_y = ends_x_y[i]
    if type(colors) is list:
      color = tuple(colors[i])
    else:
      color = colors
    im_with_arrow = cv2.arrowedLine(cv2.UMat(im_with_arrow), tuple(np.array(origin_x_y, dtype='int')), tuple(np.array(end_x_y, dtype='int')), color, width).get()

  if type(im_with_arrow) is np.ndarray:
    return im_with_arrow.transpose((2, 0, 1))
  else:
    return np.array(im_with_arrow.get()).transpose((2, 0, 1))

def tqdm_enumerate(what):
  return enumerate(tqdm(what))

def text_editor():
  txt = 'This is a write demo notepad. Type below. Delete clears text:<br>'
  callback_text_window = global_vis.text(txt, win='asdf')

  def type_callback(event):
    if event['event_type'] == 'KeyPress':
      curr_txt = event['pane_data']['content']
      if event['key'] == 'Enter':
        curr_txt += '<br>'
      elif event['key'] == 'Backspace':
        curr_txt = curr_txt[:-1]
      elif event['key'] == 'Delete':
        curr_txt = txt
      elif len(event['key']) == 1:
        curr_txt += event['key']
      global_vis.text(curr_txt, win=callback_text_window)

  global_vis.register_event_handler(type_callback, 'asdf')

def line_selector_x_y(img, window):
  raw_img = np.array(imshow(img, window=window, return_image=True)*255, dtype='uint8')
  #global_vis.text(txt, win=window)
  global first, second, first_finished, everything_finished
  first = np.array((0,0))
  second = None
  img_to_show = add_circle(raw_img, first, radius=1)
  imshow(img_to_show, window=window, return_image=True)
  first_finished = everything_finished = False
  def type_callback(event):
    global first, second, first_finished, everything_finished
    if event['event_type'] == 'KeyPress':
      if not first_finished:
        actual = first
      else:
        actual = second
      if event['key'] == 'ArrowUp':
        actual[1] = max(0, actual[1]-1)
      elif event['key'] == 'ArrowDown':
        actual[1] = min(raw_img.shape[1], actual[1]+1)
      elif event['key'] == 'ArrowLeft':
        actual[0] = max(0, actual[0]-1)
      elif event['key'] == 'ArrowRight':
        actual[0] = min(raw_img.shape[2], actual[0]+1)
      elif event['key'] == ' ':
        if first_finished:
          everything_finished = True
        else:
          first_finished = True
          second = np.array(first)
      else:
        return
      if not first_finished:
        img_to_show = add_circle(raw_img, first, radius=1)
      else:
        img_to_show = add_line(raw_img, first, second)
        img_to_show = add_circle(img_to_show, second, radius=1)

      imshow(img_to_show, window=window)

  global_vis.register_event_handler(type_callback, window)
  while not everything_finished:
    time.sleep(0.02)
  return first, second

def add_axis_to_image(im):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if len(im.shape) == 3 and im.shape[0] == 1:
    ax.imshow(im[0])
  else:
    ax.imshow(im)
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  data = data.transpose((2,0,1))
  return data

def preprocess_im_to_plot(im, normalize_image=True):
  if type(im) is list:
    for k in range(len(im)):
      im[k] = tonumpy(im[k])
    im = np.array(im)
  if type(im) == 'string':
    # it is a path
    pic = Image.open(im)
    im = np.array(pic, dtype='float32')
  im = tonumpy(im)
  if im.dtype == bool:
    im = im * 1.0
  if im.dtype == 'uint8':
    im = im / 255.0
  if len(im.shape) > 4:
    raise Exception('Im has more than 4 dims')
  if len(im.shape) == 4 and im.shape[0] == 1:
    im = im[0, :, :, :]
  if len(im.shape) == 3 and im.shape[-1] in [1, 3]:
    # put automatically channel first if its last
    im = im.transpose((2, 0, 1))
  if len(im.shape) == 2:
    # expand first if 1 channel image
    im = im[None, :, :]
  range_min, range_max = im.min(), im.max()

  if normalize_image and im.max() != im.min():
    im = (im - im.min()) / (im.max() - im.min())
  return im, range_min, range_max

def imshow(im, title='none', path=None, biggest_dim=None, normalize_image=True,
           max_batch_display=10, window=None, env=None, fps=10, vis=None,
           add_ranges=False, return_image=False, add_axis=False, gif=False, verbosity=0):
  # If video generation fails, install ffmpeg with conda:
  # https://anaconda.org/conda-forge/ffmpeg
  # conda install -c conda-forge ffmpeg

  if env is None:
    env = PYCHARM_VISDOM
  if window is None:
    window = title

  im, range_min, range_max = preprocess_im_to_plot(im, normalize_image)
  postfix = ''
  if add_ranges:
    postfix = '_max_{:.2f}_min_{:.2f}'.format()
  if not biggest_dim is None and len(im.shape) == 3:
    im = scale_image_biggest_dim(im, biggest_dim)

  if add_axis:
    if len(im.shape) == 3:
      im = add_axis_to_image(im)
    else:
      for k in range(len(im)):
        im[k] = add_axis_to_image(im[k])

  if path is None:
    if window is None:
      window = title
    if len(im.shape) == 4:
      if not gif:
        return vidshow_vis(im, title=title, window=window, env=env, vis=vis, biggest_dim=biggest_dim, fps=fps, verbosity=verbosity)
      else:
        temp_name = '{}/{}.gif'.format(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))
        make_gif(im, path=temp_name, fps=fps, biggest_dim=biggest_dim)
        return vidshow_gif_path(temp_name, title=title, win=window, env=env, vis=vis)
    else:
      imshow_vis(im, title=title + postfix, win=window, env=env, vis=vis)
  else:
    if len(im.shape) == 4:
      make_gif(im, path=path, fps=fps, biggest_dim=biggest_dim)
    else:
      imshow_matplotlib(im, path)
  if return_image:
    return im

def make_gif(ims, path, fps=None, biggest_dim=None):
  if ims.dtype != 'uint8':
    ims = np.array(ims*255, dtype='uint8')
  if ims.shape[1] in [1,3]:
    ims = ims.transpose((0,2,3,1))
  if ims.shape[-1] == 1:
    ims = np.tile(ims, (1,1,1,3))
  with imageio.get_writer(path) as gif_writer:
    for k in range(ims.shape[0]):
      #imsave(ims[k].mean()
      if biggest_dim is None:
        actual_im = ims[k]
      else:
        actual_im = np.transpose(scale_image_biggest_dim(np.transpose(ims[k]), biggest_dim))
      gif_writer.append_data(actual_im)
  if not fps is None:
    gif = imageio.mimread(path)
    imageio.mimsave(path, gif, fps=fps)

def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list


def find_all_files_recursively(folder, prepend_folder=False, extension=None, progress=False, substring=None, include_folders=False, max_n_files=-1):
  if extension is None:
    glob_expresion = '*'
  else:
    glob_expresion = '*' + extension
  all_files = []
  for f in Path(folder).rglob(glob_expresion):
    if max_n_files > 0 and len(all_files) >= max_n_files:
      return all_files
    file_name = str(f) if prepend_folder else f.name
    if substring is None or substring in file_name:
      if include_folders or not os.path.isdir(file_name):
        all_files.append(file_name)
        if progress and len(all_files) % 1000 == 0:
          print("Found {} files".format(len(all_files)))
  return all_files

def interlace(list_of_lists):
  #all elements same length
  assert len(set([len(k) for k in list_of_lists])) <= 1
  interlaced_list = [None]*len(list_of_lists)*len(list_of_lists[0])
  for k in range(len(list_of_lists)):
    act_list = list_of_lists[k]
    interlaced_list[k::len(list_of_lists)] = act_list
  return interlaced_list

def float2str(float, prec=2):
  return ("{0:." + str(prec) + "f}").format(float)


def str2img(string_to_print, height=100, width=100):
  img = Image.new('RGB', (width, height))
  d = ImageDraw.Draw(img)
  d.text((20, 20), string_to_print, fill=(255, 255, 255))
  return np.array(img).transpose((2,0,1))

def count_trainable_parameters(network, return_as_string=False):
  n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
  if return_as_string:
    return f"{n_parameters:,}"
  else:
    return n_parameters

import math
millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def beautify_integer(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def create_plane_pointcloud_coords(center, normal, extent, samples, color=(0,0,0)):
  if normal[0] == 0 and normal[1] == 0:
    #handle special case where the perpendicular cannot have 0 z component
    dir_1 = np.array((0,1,0))
  else:
    dir_1 = np.array((-normal[1],normal[0],0))
  dir_2 = np.cross(normal, dir_1)
  dir_1 = dir_1/np.linalg.norm(dir_1)
  dir_2 = dir_2 / np.linalg.norm(dir_2)
  a, b = np.mgrid[-0.5:0.5:complex(samples), -0.5:0.5:complex(samples)]
  coords = np.array(center)[:,None, None] + a[None,:,:] * dir_1[:,None, None] * extent + b[None,:,:] * dir_2[:,None, None] * extent
  coords = coords.reshape((3,-1)).transpose()
  colors = np.array([color]*coords.shape[0])
  return coords, colors

def generate_bbox_coords(min_corner, max_corner, use_max_distance=True):
  x_min, y_min, z_min = min_corner
  x_max, y_max, z_max = max_corner
  x_dist_bbox = (x_max - x_min)
  y_dist_bbox = (y_max - y_min)
  z_dist_bbox = (z_max - z_min)
  bbox_center = np.array((x_dist_bbox / 2 + x_min, y_dist_bbox / 2 + y_min, z_dist_bbox / 2 + z_min))
  if use_max_distance:
    max_bbox_dist = np.array((x_dist_bbox, y_dist_bbox, z_dist_bbox)).max() / 2
  bbox_coords = list()
  for to_bits in range(8):
    i = int(to_bits / 4)
    j = int(to_bits / 2)
    k = int(to_bits % 2)
    #first 4 with y negative (floor bbox) final 4 y positive (top bbox)
    if use_max_distance:
      bbox_coords.append(bbox_center + (max_bbox_dist * (-1) ** j, max_bbox_dist * (-1) ** (i + 1), max_bbox_dist * (-1) ** k))
    else:
      bbox_coords.append(bbox_center + (x_dist_bbox/2.0 * (-1) ** j, y_dist_bbox/2.0 * (-1) ** (i + 1), z_dist_bbox/2.0 * (-1) ** k))
  return bbox_coords


try:
  default_side_colors = np.array(sns.color_palette("hls", 12)) * 255.0
  default_corner_colors = (np.array(sns.color_palette("hls", 8)) * 255.0).transpose()
except:
  pass

def create_bbox_sides_and_corner_coords(min_corner, max_corner, bbox_center, bbox_rotation_matrix, coords_per_side = 1000, color=None):
  all_coords = list()

  ones = np.ones((1, coords_per_side))
  x_range = np.mgrid[min_corner[0]:max_corner[0]:complex(coords_per_side)][None,:]
  all_coords.append(np.concatenate((x_range, min_corner[1]*ones, min_corner[2]*ones)))
  all_coords.append(np.concatenate((x_range, min_corner[1]*ones, max_corner[2]*ones)))
  all_coords.append(np.concatenate((x_range, max_corner[1]*ones, min_corner[2]*ones)))
  all_coords.append(np.concatenate((x_range, max_corner[1]*ones, max_corner[2]*ones)))

  y_range = np.mgrid[min_corner[1]:max_corner[1]:complex(coords_per_side)][None,:]
  all_coords.append(np.concatenate((min_corner[0]*ones, y_range, min_corner[2]*ones)))
  all_coords.append(np.concatenate((min_corner[0]*ones, y_range, max_corner[2]*ones)))
  all_coords.append(np.concatenate((max_corner[0]*ones, y_range, min_corner[2]*ones)))
  all_coords.append(np.concatenate((max_corner[0]*ones, y_range, max_corner[2]*ones)))

  z_range = np.mgrid[min_corner[2]:max_corner[2]:complex(coords_per_side)][None,:]
  all_coords.append(np.concatenate((min_corner[0]*ones, min_corner[1]*ones, z_range)))
  all_coords.append(np.concatenate((min_corner[0]*ones, max_corner[1]*ones, z_range)))
  all_coords.append(np.concatenate((max_corner[0]*ones, min_corner[1]*ones, z_range)))
  all_coords.append(np.concatenate((max_corner[0]*ones, max_corner[1]*ones, z_range)))

  corner_coords = np.array(generate_bbox_coords(min_corner, max_corner, use_max_distance=False)).transpose()

  all_coords = np.array(all_coords)
  all_coords_sides_shape = all_coords.shape

  all_coords = all_coords.transpose((1,0,2)).reshape(3,-1)
  if color is None:
    current_palette = default_side_colors
    all_colors = np.ones(all_coords_sides_shape)*current_palette[:,:,None]
    all_colors = all_colors.reshape(3,-1)
    corner_colors = default_corner_colors
  else:
    all_colors = np.ones((all_coords.shape))*np.array(color)[:,None]
    corner_colors = np.ones((3,8))*np.array(color)[:,None]


  all_coords = np.matmul(bbox_rotation_matrix, all_coords)
  all_coords = all_coords + bbox_center[:,None]

  corner_coords = np.matmul(bbox_rotation_matrix, corner_coords)
  corner_coords = corner_coords + bbox_center[:, None]


  return all_coords, all_colors, corner_coords, corner_colors

def add_camera_to_pointcloud(position, rotation, focal_deg_vert, width, height, coords, np_colors=None, n_points_per_side=100, camera_color=(0,0,0)):
  scale = coords[:,2].max()*0.1
  all_coords = []
  focal_deg_horiz = focal_deg_vert*width/height
  for z in np.arange(0, 1, 1.0/n_points_per_side):
    # camera cone
    all_coords.append(np.matmul(xrotation_deg(focal_deg_vert/2), np.matmul(yrotation_deg(focal_deg_horiz/2), scale*np.array((0,0,z)))))
    all_coords.append(np.matmul(xrotation_deg(-focal_deg_vert/2), np.matmul(yrotation_deg(focal_deg_horiz/2), scale*np.array((0,0,z)))))
    all_coords.append(np.matmul(xrotation_deg(focal_deg_vert/2), np.matmul(yrotation_deg(-focal_deg_horiz/2), scale*np.array((0,0,z)))))
    all_coords.append(np.matmul(xrotation_deg(-focal_deg_vert/2), np.matmul(yrotation_deg(-focal_deg_horiz/2), scale*np.array((0,0,z)))))

  corner_points = all_coords[-4:]
  top_side_coords = np.concatenate((np.mgrid[corner_points[2][0]:corner_points[0][0]:complex(n_points_per_side)][:,None],
                                   np.ones((n_points_per_side,1))*corner_points[1][1],
                                   np.ones((n_points_per_side,1))*corner_points[0][2]), axis=1)
  bottom_side_coords = np.concatenate((np.mgrid[corner_points[2][0]:corner_points[0][0]:complex(n_points_per_side)][:,None],
                                   np.ones((n_points_per_side,1))*corner_points[0][1],
                                   np.ones((n_points_per_side,1))*corner_points[0][2]), axis=1)
  left_side_coords = np.concatenate((np.ones((n_points_per_side,1))*corner_points[2][0],
                                   np.mgrid[corner_points[0][1]:corner_points[1][1]:complex(n_points_per_side)][:,None],
                                   np.ones((n_points_per_side,1))*corner_points[0][2]), axis=1)
  right_side_coords = np.concatenate((np.ones((n_points_per_side,1))*corner_points[0][0],
                                   np.mgrid[corner_points[0][1]:corner_points[1][1]:complex(n_points_per_side)][:,None],
                                   np.ones((n_points_per_side,1))*corner_points[0][2]), axis=1)
  all_coords.extend(top_side_coords)
  all_coords.extend(bottom_side_coords)
  all_coords.extend(left_side_coords)
  all_coords.extend(right_side_coords);

  all_coords = np.array(all_coords)
  if not rotation is None:
    all_coords = np.matmul(rotation, all_coords.transpose()).transpose()
  if not position is None:
    all_coords = all_coords + position[None,:]
  if np_colors is None:
    return np.concatenate((coords, all_coords))
  else:
    new_colors = np.array([camera_color]*len(all_coords), dtype='uint8')
    return np.concatenate((coords, all_coords)), np.concatenate((np_colors, new_colors))


def create_camera(height, width, fov_x_deg=None):
  actual_camera = o3d.camera.PinholeCameraParameters()
  if fov_x_deg is None:
    fx = width / 2
    fy = width / 2
    cx = width / 2 - 0.5
    cy = height / 2 - 0.5
  else:
    intrinsics_mat = fov_x_to_intrinsic_deg(height=height, width=width, fov_x_deg=fov_x_deg, return_inverse=False)
    fx = intrinsics_mat[0,0]
    fy = intrinsics_mat[1,1]
    cx = intrinsics_mat[0,2]
    cy = intrinsics_mat[1,2]
  actual_camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

  return actual_camera


def create_static_trajectory(init_pos, height, width, samples = 100):
  cams = []
  for k in range(samples):
    actual_camera = create_camera(height, width)

    extrinsic = np.zeros((4,4))
    extrinsic[:3,:3] = xrotation_deg(-30*k/samples)
    extrinsic[:3,3] = init_pos + np.array((0, 0.5*k/samples, 0))
    extrinsic[3, 3] = 1
    actual_camera.extrinsic = extrinsic
    cams.append(actual_camera)
  for k in range(samples):
    actual_camera = create_camera(height, width)

    extrinsic = np.zeros((4,4))
    extrinsic[:3,:3] = xrotation_deg(-30*(samples - k - 1)/samples)
    extrinsic[:3,3] = init_pos + np.array((0, 0.5*(samples - k - 1)/samples, 0))
    extrinsic[3, 3] = 1
    actual_camera.extrinsic = extrinsic
    cams.append(actual_camera)

  trajectory = o3d.camera.PinholeCameraTrajectory()
  trajectory.parameters = cams

  return trajectory

default_camera_trajectory_file = 'normals_and_height/visualization/default_trajectory.json'
def create_default_trajectory(init_pos, height, width, samples=100):
  if os.path.exists(default_camera_trajectory_file):
    trajectory = o3d.io.read_pinhole_camera_trajectory(default_camera_trajectory_file)
    # remove static things
    final_trajectory_cams = []
    for k in range(len(trajectory.parameters) - 1):
      extrinsic_diff = np.abs((trajectory.parameters[k].extrinsic - trajectory.parameters[k + 1].extrinsic)).sum()
      if extrinsic_diff > 0.01:
        final_trajectory_cams.append(trajectory.parameters[k])
    trajectory.parameters = final_trajectory_cams
  else:
    cams = []
    for k in range(samples):
      actual_camera = create_camera(height, width)

      extrinsic = np.zeros((4,4))
      extrinsic[:3,:3] = xrotation_deg(-30*k/samples)
      extrinsic[:3,3] = init_pos + np.array((0, 0.5*k/samples, 0))
      extrinsic[3, 3] = 1
      actual_camera.extrinsic = extrinsic
      cams.append(actual_camera)

    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = cams

  return trajectory

def open3d_translation_transform(translation):
  assert len(translation.shape) == 1 and translation.shape[0] == 3

  transform = np.eye(4)
  transform[:3,3] = translation

  return transform

def transform_to_o3d_trajectory(trajectory, height, width, example_trajectory):
  cams = []
  for params in trajectory:
    actual_camera = create_camera(height, width) #, fov_x_deg=params['h_fov_deg'])

    extrinsic = np.zeros((4,4))
    yaw_deg, pitch_deg, roll_deg = np.rad2deg(params['yaw_rad']), np.rad2deg(params['pitch_rad']), np.rad2deg(params['roll_rad'])
    R = zrotation_deg(roll_deg + 180) @ xrotation_deg(pitch_deg) @ yrotation_deg(yaw_deg + 180)
    extrinsic[:3,:3] = R
    # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
    C = np.array((params['pos'][0], params['height'], -1*params['pos'][1]))
    T = -1 * R @ C[:, None]
    extrinsic[:3,3] = T[:,0]
    extrinsic[3, 3] = 1
    actual_camera.extrinsic = extrinsic
    cams.append(actual_camera)

  trajectory = o3d.camera.PinholeCameraTrajectory()
  trajectory.parameters = cams

  return trajectory

def draw_properties(trajectory, pcd, video_writer=None, background_color=(129, 125, 125),
                    render_height=1080, render_width=1920, axis=False,
                    break_after_completion=True, meshify=True, add_y_0_plane=False):
  draw_properties.index = -1
  initial_pos = np.array((0,0.7,0))

  example_trajectory = create_default_trajectory(initial_pos, render_height, render_width)
  draw_properties.trajectory = transform_to_o3d_trajectory(trajectory, render_height, render_width, example_trajectory)
  draw_properties.vis = o3d.visualization.Visualizer()
  images = []
  video_writer = video_writer

  def move_forward(vis, y_0_plane):
    # This function is called within the o3d.visualization.Visualizer::run() loop
    # The run loop calls the function, then re-render
    # So the sequence in this function is to:
    # 1. Capture frame
    # 2. index++, check ending criteria
    # 3. Set camera
    # 4. (Re-render)
    ctr = vis.get_view_control()
    glb = draw_properties
    if glb.index >= 0:
      image = vis.capture_screen_float_buffer(False)
      current_frame = np.array(image)
      if video_writer is None:
        images.append(current_frame)
      else:
         video_writer.writeFrame(np.array(current_frame*255, dtype='uint8'))
    glb.index = glb.index + 1
    if not y_0_plane is None:
      total_steps = 40
      if glb.index % total_steps == 0:
        vis.remove_geometry(y_0_plane)
      if glb.index % total_steps == total_steps // 2:
        vis.add_geometry(y_0_plane)
    if glb.index < len(glb.trajectory.parameters):
      ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
    else:
      if break_after_completion:
        draw_properties.vis.register_animation_callback(None)
        raise Exception("Finished!")
    return False

  vis = draw_properties.vis
  vis.create_window(width=render_width, height=render_height, left=0, top=0)
  if meshify:
    pcd_with_normals = o3d.geometry.PointCloud(pcd)
    pcd_with_normals.estimate_normals()
    pcd_with_normals.orient_normals_towards_camera_location(np.array((0,1,0)))
    #orient_normals_towards_camera_location(())
    radius = 0.03
    mesh_from_pcd = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_with_normals,
                                                                                    o3d.utility.DoubleVector([radius, radius * 2]))
    vis.add_geometry(mesh_from_pcd)
    vis.add_geometry(pcd)
  else:
    vis.add_geometry(pcd)
  if axis:
    mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(mesh_coord_frame)
  if add_y_0_plane:
    size = 25
    floor = o3d.geometry.TriangleMesh.create_box(width=size, height=0.01, depth=size)
    floor.paint_uniform_color([0.4, 0.4, 0.4])

    floor.transform(open3d_translation_transform(np.array((-size/2, 0, -size/2))))
  else:
    floor = None
  render_option = vis.get_render_option()
  render_option.load_from_json("/data/vision/torralba/movies_sfm/home/Downloads/Open3D/examples/TestData/renderoption.json")
  render_option.background_color = np.array(background_color) / 255.0
  render_option.point_size = 5.0
  vis.register_animation_callback(lambda x: move_forward(x, floor))
  try:
    vis.run()
  except:
    pass
  vis.destroy_window()
  return


# modified from open3d headles_rendering example
def record_camera_trajectory(initial_pos, pcd,
                             background_color=(129, 125,125), render_height=1080, render_width=1920):
    record_camera_trajectory.index = -1
    # just the initial camera
    record_camera_trajectory.trajectory = create_default_trajectory(initial_pos, render_height, render_width)
    record_camera_trajectory.trajectory.parameters = record_camera_trajectory.trajectory.parameters[0:1]

    record_camera_trajectory.vis = o3d.visualization.Visualizer()
    images = []
    global cameras
    global default_camera_trajectory_file

    cameras = []

    def record_camera(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = record_camera_trajectory
        if glb.index >= 0:
            image = vis.capture_screen_float_buffer(False)
            images.append(np.array(image))
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
        cameras.append(ctr.convert_to_pinhole_camera_parameters())
        current_traj = o3d.camera.PinholeCameraTrajectory()
        current_traj.parameters = cameras
        o3d.io.write_pinhole_camera_trajectory(default_camera_trajectory_file, current_traj)
        return False

    vis = record_camera_trajectory.vis
    vis.create_window(width=render_width, height=render_height , left=0, top=0)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.background_color = np.array(background_color)/255.0
    render_option.load_from_json("/data/vision/torralba/movies_sfm/home/Downloads/Open3D/examples/TestData/renderoption.json")
    render_option.point_size = 5.0
    vis.register_animation_callback(record_camera)
    try:
      vis.run()
    except Exception as e:
      pass
    vis.destroy_window()
    return images


def draw_horizon_line(image, rotation_mat=None, pitch_angle=None, roll_angle=None, intrinsic=None, force_no_roll=False, color=(255,0,0)):
  #the horizon is the perpendicular of the line of the vanishing x and y axes
  if rotation_mat is None:
    roll_rotations = zrotation_deg(-1 * float(roll_angle))
    pitch_rotations = xrotation_deg(float(pitch_angle))
    rotation_mat = pitch_rotations @ roll_rotations
  assert image.dtype == 'uint8'
  height, width = image.shape[1:]
  image = np.ascontiguousarray(tonumpy(image))
  if intrinsic is None:
    intrinsic = np.array(((width, 0, width / 2.0),
                          (0, height, height / 2.0),
                          (0, 0, 1)))

  #the ground plane is the xy
  projection_mat = np.matmul(intrinsic, rotation_mat)
  vanishing_x = np.matmul(projection_mat, np.array((1,0,0)))
  vanishing_z = np.matmul(projection_mat, np.array((0,0,1)))

  if vanishing_x[2] == 0 or vanishing_z[2] == 0:
    vanishing_direction = np.array((1,0))
  else:
    vanishing_x_image_plane = vanishing_x[:2]/vanishing_x[2]
    vanishing_z_image_plane = vanishing_z[:2]/vanishing_z[2]

    vanishing_direction = vanishing_z_image_plane - vanishing_x_image_plane

  if force_no_roll:
    vanishing_direction = np.array((1,0))

  start_y = vanishing_z_image_plane[1] - vanishing_z_image_plane[1]*vanishing_direction[1]/vanishing_direction[0]
  finish_y = vanishing_z_image_plane[1] + (width - vanishing_z_image_plane[1])*vanishing_direction[1]/vanishing_direction[0]
  im = Image.fromarray(image.transpose((1,2,0)))
  draw = ImageDraw.Draw(im)
  draw.line((0, start_y, im.size[0], finish_y), fill=color, width=int(math.ceil(height/100)))
  image_with_line = np.array(im).transpose((2,0,1))
  return image_with_line


# open nvidia-xorg
# sudo apt-get install xserver-xorg
# configure xorg to use nvidia
# sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# sudo /usr/bin/X :1
# then, while running the script
# sudo apt-get install x11vnc
# x11vnc -display :1
# then open vncviewer remotely

def create_video_from_pointcloud_and_trajectory(original_coords, original_colors, trajectory,
                                                video_writer=None, background_color=(255, 255, 255), display_number=1, meshify=True):
  coords, colors = prepare_single_pointcloud_and_colors(original_coords, original_colors)
  coords[:, 1] = -1 * coords[:, 1]
  coords[:, 2] = -1 * coords[:, 2]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(coords)
  if colors.dtype == 'uint8':
    colors = colors / 255.0
  #colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  try:
    os.environ['DISPLAY'] = ':' + str(display_number)
    draw_properties(trajectory, pcd, video_writer=video_writer,
                    background_color=background_color, render_height=int(1080/2), render_width=int(1920/2), meshify=meshify)
    if not video_writer is None:
      video_writer.close()
  except Exception as e:
    print("Failed to render on display :{}".format(display_number))
    print(e)
  return

def np_where_in_multiple_dimensions(true_false_array):
  indices = np.where(true_false_array.flatten())
  unraveled_indices = np.unravel_index(indices, true_false_array.shape)

  return np.array(unraveled_indices)[:,0,:].transpose()

def get_grads(network):
  params = list(network.parameters())
  grads = [p.grad for p in params]
  return grads

def scale_img_to_fit_canvas(img, canvas_height, canvas_width):
  im = Image.fromarray(img.transpose((1,2,0)))
  im.thumbnail((canvas_height, canvas_width), Image.ANTIALIAS)
  return np.array(im).transpose((2,0,1))


def create_text_image(lines, line_size=(50,500), color=(0, 0, 0)):
  im = 255*np.ones((line_size[0]*len(lines), line_size[1], 3), dtype='uint8')
  font = cv2.FONT_HERSHEY_TRIPLEX
  for k in range(len(lines)):
    im = cv2.putText(im, lines[k], (10, line_size[0]*(k + 1) - 10), font, 1, color, 2, cv2.LINE_AA)
  return im


def add_coord_map(input_tensor):
  # https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """
  Args:
      input_tensor: shape(batch, channel, x_dim, y_dim)
  """
  batch_size, _, x_dim, y_dim = input_tensor.size()

  xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
  yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

  xx_channel = xx_channel.float() / (x_dim - 1)
  yy_channel = yy_channel.float() / (y_dim - 1)

  xx_channel = xx_channel * 2 - 1
  yy_channel = yy_channel * 2 - 1

  xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
  yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

  ret = torch.cat([
    input_tensor,
    xx_channel.type_as(input_tensor),
    yy_channel.type_as(input_tensor)], dim=1)
  return ret

def rotate_pointcloud(pcl, rotation_mat):
  original_pcl_shape = pcl.shape
  assert original_pcl_shape[0] == 3
  assert rotation_mat.shape == (3,3)
  flattened_pcl = pcl.reshape((3,-1))
  return np.matmul(rotation_mat, flattened_pcl).reshape(original_pcl_shape)

def show_pointcloud_errors(coords, errors, title='none', win=None, env=None, markersize=3, max_points=10000,
                    force_aspect_ratio=True, valid_mask=None):
  if env is None:
    env = PYCHARM_VISDOM
  if type(coords) is list:
    assert type(errors) is list and type(title) is list
    assert len(coords) == len(errors) and len(coords) == len(title)
    # Normalize error colors together, so that they show the same color map
    errors = np.array(errors)
  else:
    coords = [coords]
    title = [title]
    errors = errors[None,:,:]

  assert coords[0].shape[1:] == errors[0].shape

  color = np.array((255, 0, 0))
  errors = errors.reshape((errors.shape[0], -1))
  np_error_colors = np.array((errors/errors.max())[:, None,:]*color[None,:,None], dtype='uint8')

  for k in range(len(coords)):
    labels = ['Error: {}'.format(str(k)) for k in errors[k].flatten()]

    show_pointcloud(coords[k].reshape((3,-1)), np_error_colors[k], title=title[k], win=win, env=env, markersize=markersize, max_points=max_points,
                      force_aspect_ratio=force_aspect_ratio, valid_mask=valid_mask, nice_plot_rotation=nice_plot_rotation, labels=labels)


def prepare_pointclouds_and_colors(coords, colors, default_color=(0,0,0), max_points=-1):
  if type(coords) is list:
    coords = [k for k in coords]
    if not colors is None:
      colors = [k for k in colors]
    processed_colors = None
    for k in range(len(coords)):
      if not colors is None:
        if processed_colors is None:
          processed_colors = []
        assert len(coords) == len(colors)
        colors_to_proc = colors[k]
      else:
        colors_to_proc = None
      coords[k], cur_colors = prepare_single_pointcloud_and_colors(coords[k], colors_to_proc, default_color)
      if max_points > 0 and len(coords) > max_points:
        points_selection = random.sample(range(coords[k].shape[0]), max_points)
        coords[k] = coords[k][points_selection]
      else:
        points_selection = None
      if not colors is None:
        if not points_selection is None:
          cur_colors = cur_colors[points_selection]
        processed_colors.append(cur_colors)
    return coords, processed_colors
  else:
    return prepare_single_pointcloud_and_colors(coords, colors, default_color)

def prepare_single_pointcloud_and_colors(coords, colors, default_color=(0,0,0)):
  coords = tonumpy(coords)
  if colors is None:
    colors = np.array(default_color)[:, None].repeat(coords.size / 3, 1).reshape(coords.shape)
  colors = tonumpy(colors)
  if colors.dtype == 'float32':
    if colors.max() > 1.0:
      colors = np.array(colors, dtype='uint8')
    else:
      colors = np.array(colors * 255.0, dtype='uint8')
  if type(coords) is list:
    for k in range(len(colors)):
      if colors[k] is None:
        colors[k] = np.ones(coords[k].shape)
    colors = np.concatenate(colors, axis=0)
    coords = np.concatenate(coords, axis=0)

  assert coords.shape == colors.shape
  if len(coords.shape) == 3:
    coords = coords.reshape((3, -1))
  if len(colors.shape) == 3:
    colors = colors.reshape((3, -1))
  assert len(coords.shape) == 2
  if coords.shape[0] == 3:
    coords = coords.transpose()
    colors = colors.transpose()
  return coords, colors

def get_plane_pointcloud(plane_params, x_extension=None, y_extension=None, z_extension=None, color=(0,0,0), n_points=100):
  assert ((x_extension is None) + (y_extension is None) + (z_extension is None)) == 1
  if x_extension is None:
    y, z = np.mgrid[y_extension[0]:y_extension[1]:complex(n_points), z_extension[0]:z_extension[1]:complex(n_points)]
    x = -(y*plane_params[1] + z*plane_params[2] + plane_params[3])/plane_params[0]
  elif y_extension is None:
    z, x = np.mgrid[z_extension[0]:z_extension[1]:complex(n_points), x_extension[0]:x_extension[1]:complex(n_points)]
    y = -(x * plane_params[0] + z * plane_params[2] + plane_params[3]) / plane_params[1]
  else:
    y, x = np.mgrid[y_extension[0]:y_extension[1]:complex(n_points), x_extension[0]:x_extension[1]:complex(n_points)]
    z = -(x * plane_params[0] + y * plane_params[1] + plane_params[3]) / plane_params[2]

  points = np.concatenate((x[None,:,:], y[None,:,:], z[None,:,:])).reshape(3, -1)
  colors = np.array([np.array(color)]*points.shape[1]).transpose()
  return points, colors


def show_pointcloud(original_coords, original_colors=None, title='none', win=None, env=None,
                    markersize=3, max_points=10000, valid_mask=None, labels=None, default_color=(0,0,0),
                    projection="orthographic", center=(0,0,0), up=(0,-1,0), eye=(0,0,-2),
                    display_grid=(True,True,True), axis_ranges=None):
  if env is None:
    env = PYCHARM_VISDOM
  assert projection in ["perspective", "orthographic"]
  coords, colors = prepare_pointclouds_and_colors(original_coords, original_colors, default_color, max_points=max_points)
  if not type(coords) is list:
    coords = [coords]
    colors = [colors]
  if not valid_mask is None:
    if not type(valid_mask) is list:
      valid_mask = [valid_mask]
    assert len(valid_mask) == len(coords)
    for i in range(len(coords)):
      if valid_mask[i] is None:
        continue
      else:
        actual_valid_mask = np.array(valid_mask[i], dtype='bool').flatten()
        coords[i] = coords[i][actual_valid_mask]
        colors[i] = colors[i][actual_valid_mask]
  if not labels is None:
    if not type(labels) is list:
      labels = [labels]
    assert len(labels) == len(coords)
  for i in range(len(coords)):
    if max_points != -1 and coords[i].shape[0] > max_points:
      selected_positions = random.sample(range(coords[i].shape[0]), max_points)
      coords[i] = coords[i][selected_positions]
      colors[i] = colors[i][selected_positions]
      if not labels is None:
        labels[i] = [labels[i][k] for k in selected_positions]
      if not type(markersize) is int or type(markersize) is float:
        markersize[i] = [markersize[i][k] for k in selected_positions]
  # after this, we can compact everything into a single set of pointclouds. and do some more stuff for nicer visualization
  coords = np.concatenate(coords)
  colors = np.concatenate(colors)
  if not type(markersize) is int or type(markersize) is float:
    markersize = list_of_lists_into_single_list(markersize)
    assert len(coords) == len(markersize)
  if not labels is None:
    labels = list_of_lists_into_single_list(labels)
  if win is None:
    win = title
  plot_coords = coords
  from visdom import _markerColorCheck
  # we need to construct our own colors to override marker plotly options
  # and allow custom hover (to show real coords, and not the once used for visualization)
  visdom_colors = _markerColorCheck(colors, plot_coords, np.ones(len(plot_coords), dtype='uint8'), 1)
  # add the coordinates as hovertext
  hovertext = ['x:{:.2f}\ny:{:.2f}\nz:{:.2f}\n'.format(float(k[0]), float(k[1]), float(k[2])) for k in coords]
  if not labels is None:
    assert len(labels) == len(hovertext)
    hovertext = [hovertext[k] + ' {}'.format(labels[k]) for k in range(len(hovertext))]

  # to see all the options interactively, click on edit plot on visdom->json->tree
  # traceopts are used in line 1610 of visdom.__intit__.py
  # data.update(trace_opts[trace_name])
  # for layout options look at _opts2layout

  camera = {'up':{
              'x': str(up[0]),
              'y': str(up[1]),
              'z': str(up[2]),
            },
            'eye':{
              'x': str(eye[0]),
              'y': str(eye[1]),
              'z': str(eye[2]),
            },
            'center':{
              'x': str(center[0]),
              'y': str(center[1]),
              'z': str(center[2]),
            },
            'projection': {
              'type': projection
            }
          }

  global_vis.scatter(plot_coords, env=env, win=win,
              opts={'webgl': True,
                    'title': title,
                    'name': 'scatter',
                    'layoutopts': {
                      'plotly':{
                        'scene': {
                          'aspectmode': 'data',
                          'camera': camera,
                          'xaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_x']), str(axis_ranges['max_x'])] if not axis_ranges is None else [-1,-1],
                            'showgrid': display_grid[0],
                            'showticklabels': display_grid[0],
                            'zeroline': display_grid[0],
                            'title': {
                                  'text':'x' if display_grid[0] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          },
                          'yaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_y']), str(axis_ranges['max_y'])] if not axis_ranges is None else [-1, -1],
                            'showgrid': display_grid[1],
                            'showticklabels': display_grid[1],
                            'zeroline': display_grid[1],
                            'title': {
                                  'text':'y' if display_grid[1] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          },
                          'zaxis': {
                            'tickfont':{
                              'size': 14
                            },
                            'autorange': axis_ranges is None,
                            'range': [str(axis_ranges['min_z']), str(axis_ranges['max_z'])] if not axis_ranges is None else [-1, -1],
                            'showgrid': display_grid[2],
                            'showticklabels': display_grid[2],
                            'zeroline': display_grid[2],
                            'title': {
                                  'text':'z' if display_grid[2] else '',
                                  'font':{
                                    'size':20
                                    }
                                  }
                          }
                        }
                      }
                    },
                    'traceopts': {
                      'plotly':{
                        '1': {
                          #custom ops
                          # https://plot.ly/python/reference/#scattergl-transforms
                          'hoverlabel':{
                            'bgcolor': '#000000'
                          },
                          'hoverinfo': 'text',
                          'hovertext': hovertext,
                          'marker': {
                            'sizeref': 1,
                            'size': markersize,
                            'symbol': 'dot',
                            'color': visdom_colors[1],
                            'line': {
                                'color': '#0z00000',
                                'width': 0,
                            }
                          }
                        },
                      }
                    }
                  })

  return

def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files

def print_float(number):
  return "{:.2f}".format(number)

def imshow_matplotlib(im, path):
  imwrite(path,np.transpose(im, (1, 2, 0)))


def all_labels(array, nbins=20, legend=None):
  import matplotlib.pyplot as pyplt
  if type(array) == list:
    array = np.asarray(array)
  else:
    array = np.expand_dims(array,0)
  binwidth = (np.max(array) - np.min(array))/nbins
  bins = np.arange(np.min(array), np.max(array) + binwidth, binwidth)
  for i in range(array.shape[0]):
    pyplt.hist(array[i,:], bins)
  if not legend is None:
    pyplt.legend(legend)

  pyplt.xlabel("Value")
  pyplt.ylabel("Frequency")

  tmp_fig_file = '/tmp/histogram.png'
  pyplt.savefig(tmp_fig_file, bbox_inches='tight', pad_inches=0)

  image = scipy_misc.imread(tmp_fig_file)
  pyplt.close()
  return np.transpose(image,[2,0,1])

def project_points_to_plane(data_points, p_n_0):
  if data_points.shape[0] != 3:
    data_points = data_points.transpose()
    transposed = True
  else:
    transposed = False
  ls = data_points
  p_n = p_n_0[:3]
  p_0 = np.array((0,0,p_n_0[3]))
  ds = (p_0 * p_n).sum(0) / (ls * p_n[:, None]).sum(0)
  points_in_plane = ls * ds[None, :]
  if transposed:
    points_in_plane = points_in_plane.transpose()
  return points_in_plane


def fit_plane_np(data_points, robust=False):
  assert data_points.shape[0] == 3
  X = data_points.transpose()
  y = -1*np.ones(data_points.shape[1])
  if robust:
    # The following is the as the least squares solution:
    # linear = linear_model.LinearRegression(fit_intercept=False)
    # linear.fit(X,y)
    # C2 = linear.coef_

    # but we use ransac, with the same estimator
    from sklearn import linear_model

    base_estimator = linear_model.LinearRegression(fit_intercept=False)
    ransac = linear_model.RANSACRegressor(base_estimator=base_estimator, min_samples=50, )
    ransac.fit(X, y)
    C0 = base_estimator.fit(X, y).coef_
    C = ransac.estimator_.coef_
  else:
    C, _, _, _ = lstsq(X, y)
  # The new z will be the z where the original directions intersect the plane C
  p_n_0 = np.array((C[0], C[1], C[2], 1))
  return p_n_0

def create_legend_classes(class_names, class_colors, class_ids, image=None):
  from matplotlib.patches import Rectangle
  from matplotlib.gridspec import GridSpec

  gs = GridSpec(6, 1)

  fig = pyplt.figure(figsize=(6, 6))
  if not image is None:
    ax1 = fig.add_subplot(gs[:-1, :])  ##for the plot
  ax2 = fig.add_subplot(gs[-1, :])  ##for the legend
  if not image is None:
    ax1.imshow(image)

  legend_data = [[class_ids[k], class_colors[k], class_names[k]] for k in range(len(class_names))]
  handles = [
    Rectangle((0, 0), 1, 1, color=tuple((v / 255.0 for v in c))) for k, c, n in legend_data
  ]
  labels = [n for k, c, n in legend_data]

  ax2.legend(handles, labels, mode='expand', ncol=3)
  ax2.axis('off')

  tmp_fig_file = '/tmp/legend.png'
  pyplt.savefig(tmp_fig_file, bbox_inches='tight', pad_inches=0)

  image = scipy_misc.imread(tmp_fig_file)

  return image

def intrinsics_to_fov_x_deg(intrinsics):
  # assumes zero centered
  width = intrinsics[0,2] * 2
  fov_x_rad = 2 * np.arctan(width / 2.0 / intrinsics[0, 0])
  fov_x_deg = np.rad2deg(fov_x_rad)
  return fov_x_deg

def intrinsics_to_fov_y_deg(intrinsics):
  # assumes zero centered
  height = intrinsics[1,2] * 2
  fov_y_rad = 2 * np.arctan(height / 2.0 / intrinsics[1, 1])
  fov_y_deg = np.rad2deg(fov_y_rad)
  return fov_y_deg

def fov_x_to_intrinsic_deg(fov_x_deg, width, height, return_inverse=True):
  fov_y_rad = fov_x_deg / 180.0 * np.pi
  return fov_x_to_intrinsic_rad(fov_y_rad, width, height, return_inverse=return_inverse)


def fov_x_to_intrinsic_rad(fov_x_rad, width, height, return_inverse=True):
  if type(fov_x_rad) is torch.Tensor:
    f = width / (2 * torch.tan(fov_x_rad / 2))
    zero = torch.FloatTensor(np.zeros(fov_x_rad.shape[0]))
    one = torch.FloatTensor(np.ones(fov_x_rad.shape[0]))
    if fov_x_rad.is_cuda:
      zero = zero.cuda()
      one = one.cuda()
    intrinsics_0 = torch.stack((f, zero, zero)).transpose(0,1)
    intrinsics_1 = torch.stack((zero, f, zero)).transpose(0,1)
    intrinsics_2 = torch.stack((width/2, height/2, one)).transpose(0,1)
    intrinsics = torch.cat((intrinsics_0[:,:,None], intrinsics_1[:,:,None], intrinsics_2[:,:,None]), dim=-1)
  else:
    f = width / (2 * np.tan(fov_x_rad / 2))
    intrinsics = np.array(((f, 0, width / 2),
                         (0, f, height / 2),
                         (0,  0,         1)), dtype='float32')
  if return_inverse:
    if type(fov_x_rad) is torch.Tensor:
      return intrinsics, torch.inverse(intrinsics)
    else:
      return intrinsics, np.linalg.inv(intrinsics)
  else:
    return intrinsics

def dump_pointcloud(coords, colors, file_name, valid_mask=None, max_points=10000000, subsample_by_distance=False):
  if not valid_mask is None:
    coords = coords[valid_mask]
    colors = colors[valid_mask]
  if max_points != -1 and coords.shape[0] > max_points:
    if subsample_by_distance:
      #just get the ones whose neighbors are further away
      distances = -1*(np.sqrt(((coords[:-1] - coords[1:])**2).sum(-1)))
      distances_and_indices = list(zip(distances, range(len(distances))))
      distances_and_indices.sort()
      selected_positions = [k[1] for k in distances_and_indices[:max_points]]
    else:
      selected_positions = random.sample(range(coords.shape[0]), max_points)
    coords = coords[selected_positions]
    colors = colors[selected_positions]
  coords = tonumpy(coords)
  colors = tonumpy(colors)
  if coords.shape[0] == 3:
    coords = coords.transpose()
    colors = colors.transpose()
  data_np = np.concatenate((coords, colors), axis=1)
  tupled_data = [tuple(k) for k in data_np.tolist()]
  data = np.array(tupled_data, dtype=[('x', 'f4'),
                                     ('y', 'f4'),
                                     ('z', 'f4'),
                                     ('red', 'u1'),
                                     ('green', 'u1'),
                                     ('blue', 'u1')])

  vertex = PlyElement.describe(data, 'vertex')
  vertex.data = data
  plydata = PlyData([vertex])
  if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
  plydata.write(file_name + '.ply')

def dump_to_pickle(filename, obj):
  try:
    with open(filename, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
  except:
    with open(filename, 'wb') as handle:
      pickle.dump(obj, handle)

def load_from_pickle(filename):
  try:
    with open(filename, 'rb') as handle:
      return pickle.load(handle)
  except:
    pass
  try:
    with open(filename, 'rb') as handle:
      return  pickle.load(handle, encoding='latin1')
  except:
    pass
  try:
    with open(filename, 'rb') as handle:
      return  pickle.load( open( filename, "rb" ) )
  except:
    raise Exception('Failed to unpickle!')

def load_np_array(file_name):
  try:
    with open(file_name, 'rb') as infile:
      return pickle.load(infile)
  except:
    loaded_np = np.load(file_name)
    if type(loaded_np) == np.lib.npyio.NpzFile:
      return loaded_np['arr_0']
    return loaded_np

def save_np_array(array, filename):
  folder = '/'.join(filename.split('/')[:-1])
  if not os.path.exists(folder):
    os.makedirs(folder)
  with open(filename, 'wb') as outfile:
    pickle.dump(array, outfile, pickle.HIGHEST_PROTOCOL)

def define_and_make_dir(dir):
  os.makedirs(dir, exist_ok=True)
  return dir

def make_dir_without_file(file):
  folder, file = os.path.split(file)
  if len(folder) > 0:
    os.makedirs(folder, exist_ok=True)

def get_hash_from_numpy_array(numpy_array):
  # hash returns different hashes on each execution, see:
  # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
  # because of this we use md5
  return str(hashlib.md5(numpy_array.tobytes()).hexdigest())

def mkdir(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)
  return

def delete_file(file):
  if os.path.exists(file):
    os.remove(file)

def torch_deg2rad(angle_deg):
  return (angle_deg*np.pi)/180

def torch_load(torch_path, gpus):
  if len(gpus) == 0:
    return torch_load_on_cpu(torch_path)
  else:
    return torch.load(torch_path)

def intrinsics_form_fov(fov_angles, pixels_width):
  # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
  fov_rad = fov_angles*np.pi/180
  f = pixels_width/np.tan(fov_rad/2)/2
  intrinsics = np.array(((f, 0, pixels_width / 2),
                        (0, f, pixels_width / 2),
                        (0, 0, 1)))
  intrinsics_inv = np.linalg.inv(intrinsics)
  return intrinsics, intrinsics_inv

def torch_load_on_cpu(torch_path):
  return torch.load(torch_path, map_location=lambda x, y: x)


def create_flow_image(flow):
  #from cv2 flow tutorial:
  #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
  hsv = np.zeros((flow.shape[0], flow.shape[1],3))
  hsv[..., 1] = 255

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(np.array(hsv, dtype='uint8'), cv2.COLOR_HSV2BGR)
  return rgb

def crop_center(img, crop):
  cropy, cropx = crop
  if len(img.shape) == 3:
    _, y, x = img.shape
  else:
    y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  if len(img.shape) == 3:
    return img[:, starty:starty + cropy, startx:startx + cropx]
  else:
    return img[starty:starty + cropy, startx:startx + cropx]

def cv2_resize(image, target_height_width, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_height_width[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_height_width[::-1], interpolation=interpolation).transpose((2, 0, 1))

def get_image_x_y_coord_map(height, width):
  # returns image coord maps from x_y, from 0 to width -1, 0 to
  return

def best_centercrop_image(image, height, width, return_rescaled_size=False, interpolation=cv2.INTER_NEAREST):
  if height == -1 and width == -1:
    if return_rescaled_size:
      return image, image.shape
    return image
  image_height, image_width = image.shape[-2:]
  im_crop_height_shape = (int(height), int(image_width * height / image_height))
  im_crop_width_shape = (int(image_height * width / image_width), int(width))
  # if we crop on the height dimension, there must be enough pixels on the width
  if im_crop_height_shape[1] >= width:
    rescaled_size = im_crop_height_shape
  else:
    # crop over width
    rescaled_size = im_crop_width_shape
  resized_image = cv2_resize(image, rescaled_size, interpolation=interpolation)
  center_cropped = crop_center(resized_image, (height, width))
  if return_rescaled_size:
    return center_cropped, rescaled_size
  else:
    return center_cropped

class ImageFolderCenterCroppLoader():
  def __init__(self, folder_or_img_list, height, width, extension='jpg'):
    if type(folder_or_img_list) == list:
      self.img_list = folder_or_img_list
    elif os.path.isdir(folder_or_img_list):
      self.img_list = [folder_or_img_list + '/' + k for k in os.listdir(folder_or_img_list) if k.endswith(extension)]
    self.img_list.sort()
    self.height = height
    self.width = width

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, item):
    image = cv2_imread(self.img_list[item])
    img = best_centercrop_image(image, self.height, self.width)
    to_return = {'image': np.array(img / 255.0, dtype='float32'),
                 'path': self.img_list[item].split('/')[-1],
                 'full_path': self.img_list[item]}
    return to_return


def find_cuda_leaks(active_refs_before, active_refs_after):
  leaks = [k for k in active_refs_after if not id(k) in [id(r) for r in active_refs_before]]
  return leaks

def get_active_tensors(gpu_only=True):
  import gc
  all_tensors = []
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        tensor = obj
      elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
        tensor = obj.data
      else:
        continue

      if tensor.is_cuda or not gpu_only:
        all_tensors.append(tensor)
    except Exception as e:
      pass
  return all_tensors

def send_graph_to_device(g, device):
  # nodes
  labels = g.node_attr_schemes()
  for l in labels.keys():
    g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

  # edges
  labels = g.edge_attr_schemes()
  for l in labels.keys():
    g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

  return g

def tonumpy(tensor):
  if type(tensor) is Image:
    return np.array(tensor).transpose((2,0,1))
  if type(tensor) is list:
    return np.array(tensor)
  if type(tensor) is np.ndarray:
    return tensor
  else:
    try:
      if tensor.requires_grad:
        tensor = tensor.detach()
      if type(tensor) is torch.autograd.Variable:
        tensor = tensor.data
      if tensor.is_cuda:
        tensor = tensor.cpu()
      return tensor.detach().numpy()
    except:
      # try to cast still, for example if it's jax
      return np.array(tensor)
    
def totorch(array, device=None):
  if type(array) is torch.Tensor:
    return array
  if not type(array) is np.ndarray:
    array = np.array(array)
  array = torch.FloatTensor(array)
  if not device is None:
    array = array.to(device)
  return array

def tovariable(array):
  if type(array) == np.ndarray:
    array = totorch(array)
  return Variable(array)

def extrinsic_mat_to_pose(mat):
  return 1

def pose_to_extrinsic(mat):
  return 1

def subset_frames(get_dataset=False, fps=4):
  selected_movies = ['pulp_fiction_1994']
  selected_frames =[]
  #refered as indexes at 4 fps
  #selected_frames_4fps.extend(range(400, 500))
  #selected_frames_4fps.extend(range(2204, 2284))
  #selected_frames_4fps.extend(range(2666, 2811))
  #selected_frames_4fps.extend(range(2939, 3000))

  #selected_frames.extend(range(6*400,  6*500))
  #selected_frames.extend(range(6*2204, 6*2284))
  #selected_frames.extend(range(6*2666, 6*2811))
  selected_frames.extend(range(6*2939, 6*3000))

  if fps != 'all':
    selected_frames = [k/24*fps for k in selected_frames]

  if not get_dataset:
    return selected_frames, selected_movies
  else:
    return MovieSequenceDataset(selected_movies, split=selected_frames, fps=fps)#, width=width)

def get_essential_matrix(src_pts, tgt_pts, K):
  pts_src_norm = cv2.undistortPoints(src_pts, cameraMatrix=K, distCoeffs=None)
  pts_tgt_norm = cv2.undistortPoints(tgt_pts, cameraMatrix=K, distCoeffs=None)
  E, mask = cv2.findEssentialMat(pts_src_norm, pts_tgt_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)

  points, R, t, mask = cv2.recoverPose(E, pts_src_norm, pts_tgt_norm, mask=mask)

  M_r = np.hstack((R, t))
  M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

  P_l = np.dot(K, M_l)
  P_r = np.dot(K, M_r)
  point_4d_hom = cv2.triangulatePoints(P_l, P_r, src_pts, tgt_pts)
  point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
  point_3d = point_4d[:3, :].T

  return E, R, t


def to_cv2(image):
  return image.transpose((1, 2, 0))

def from_cv2(image):
  return image.transpose((2, 0, 1))

def filter_horizontal_sift_matches(L_pts, R_pts, sim_distance, env = None, L_img=None, R_img=None):
  vertical_distance = L_pts[0] - R_pts[0]
  gray_L_img = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY)
  gray_R_img = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY)

  if not L_img is None:
    match_img = cv2.drawMatches(
      gray_L_img, ref_kp,
      gray_R_img, tgt_kp,
      matches, gray_R_img.copy(), flags=0)
    imshow(match_img / 255.0, title='all_sift_matches', env=env, biggest_dim=1000)
  return L_pts, R_pts, sim_distance


def gaussian_blur_image(img, blur_pixels):
  if len(img.shape) == 2:
    return gaussian_filter(img, blur_pixels)
  else:
    assert img.shape[0] == 1 or img.shape[0] == 3
    return np.array([gaussian_filter(img[k, :, :], blur_pixels) for k in range(img.shape[0])], dtype='float32')

def detect_lines(img, return_line_and_edges_image=False):
  # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
  img_height = img.shape[1]
  gray = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_BGR2GRAY)

  kernel_size = 5
  blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
  low_threshold = 100
  high_threshold = 150

  edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

  rho = 1  # distance resolution in pixels of the Hough grid
  theta = np.pi / 180  # angular resolution in radians of the Hough grid
  threshold = int(40*img_height/720) # minimum number of votes (intersections in Hough grid cell)
  min_line_length = int(30*img_height/720)  # minimum number of pixels making up a line
  max_line_gap = int(4*img_height/720)  # maximum gap in pixels between connectable line segments
  line_image = np.copy(img) * 0  # creating a blank to draw lines on

  # Run Hough on edge detected image
  # Output "lines" is an array containing endpoints of detected line segments
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                      min_line_length, max_line_gap)
  if not return_line_and_edges_image:
    return lines

  for line in lines:
    for x1,y1,x2,y2 in line:
      line_image = add_line(line_image, (x1,y1), (x2,y2))
  return lines, line_image, edges

def get_sift_matches(gray_ref_img, gray_tgt_img, mask_ref_and_target=None, dist_threshold=-1, N_MATCHES=-1):
  try:
    sift = cv2.xfeatures2d.SIFT_create()
  except:
    print("Error when creating SIFT! Will use something else")
    sift = cv2.xfeatures2d.StarDetector_create()

  ref_kp, ref_desc = sift.detectAndCompute(gray_ref_img, None)
  tgt_kp, tgt_desc = sift.detectAndCompute(gray_tgt_img, None)

  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  matches = bf.match(ref_desc, tgt_desc)
  if dist_threshold > 0:
    matches = [ m for m in matches if m.distance < dist_threshold]
  # draw the top N matches
  if mask_ref_and_target is None:
    # Sort the matches in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:N_MATCHES]
  else:
    #sort by explainability and then by distance
    src_pts = [ref_kp[m.queryIdx].pt for m in matches]
    dst_pts = [tgt_kp[m.trainIdx].pt for m in matches]
    src_pts_exp = [mask_ref_and_target[0][int(p[1]), int(p[0])] for p in src_pts]
    dst_pts_exp = [mask_ref_and_target[1][int(p[1]), int(p[0])] for p in dst_pts]
    exp_per_match = [a*b for (a,b) in zip(src_pts_exp, dst_pts_exp)]
    matches_with_weight = zip(matches, exp_per_match)
    matches = sorted(matches_with_weight, key=lambda x: -x[1])
    matches = [m[0] for m in matches if m[1] > 0.5][:N_MATCHES]

  #matches = random.sample(population=matches, k=N_MATCHES)

  src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
  dst_pts = np.float32([tgt_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

  return src_pts, dst_pts, ref_kp, tgt_kp, matches

def cv_size(img):
    return tuple(img.shape[1::-1])

def compute_sift_image(L_img, R_img, mask_ref_and_target=None, make_plots=True, dist_threshold=-1, N_MATCHES=-1, env=None):
  if env is None:
    env = 'sift' + '_masked' if  not mask_ref_and_target is None else ''
  if L_img.shape[0] in [1, 3]:
    L_img = np.transpose(L_img, (1, 2, 0))
  if R_img.shape[0] in [1, 3]:
    R_img = np.transpose(R_img, (1, 2, 0))
  if len(L_img.shape) == 3 and L_img.shape[-1] == 3:
    gray_L_img = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY)
    gray_R_img = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY)
  else:
    gray_L_img = L_img
    gray_R_img = R_img
  if len(L_img.shape) == 2:
    L_img = L_img[:, :, None]
    R_img = R_img[:, :, None]

  L_pts, R_pts, ref_kp, tgt_kp, matches = get_sift_matches(gray_L_img, gray_R_img, mask_ref_and_target=None, dist_threshold=dist_threshold, N_MATCHES=N_MATCHES)
  sim_distances = [m.distance for m in matches]

  if make_plots:
    draw_sift_matches(gray_L_img, gray_R_img, matches, ref_kp, tgt_kp, sim_distances, env)
  return L_pts, R_pts, matches, sim_distances


def draw_sift_matches(L_img, R_img, matches, ref_kp, tgt_kp, sim_distances, env, show_biggest_dim=1000):
  less_than_100 = [d for d in sim_distances if d < 100]
  if L_img.shape[0] == 3:
    L_img = L_img.transpose((1,2,0))
    R_img = R_img.transpose((1,2,0))
  if len(L_img.shape) == 3 and L_img.shape[-1] == 3:
    L_img = cv2.cvtColor(L_img, cv2.COLOR_BGR2GRAY)
    R_img = cv2.cvtColor(R_img, cv2.COLOR_BGR2GRAY)

  if matches is None:
    assert len(ref_kp) == len(tgt_kp) and len(ref_kp) == len(sim_distances)
    matches = list()
    for k in range(len(ref_kp)):
      match = cv2.DMatch()
      match.distance = sim_distances[k]
      match.trainIdx = k
      match.queryIdx = k
      matches.append(match)

  if type(ref_kp) is np.ndarray:
    transformed_ref_kps = list()
    transformed_tgt_kps = list()
    for k in range((len(ref_kp))):
      transformed_ref_kps.append(cv2.KeyPoint(ref_kp[k][0], ref_kp[k][1], 0))
    for k in range((len(tgt_kp))):
      transformed_tgt_kps.append(cv2.KeyPoint(tgt_kp[k][0], tgt_kp[k][1], 0))
    ref_kp = transformed_ref_kps
    tgt_kp = transformed_tgt_kps

  match_img = cv2.drawMatches(
    L_img, ref_kp,
    R_img, tgt_kp,
    matches, R_img.copy(), flags=0)
  imshow(match_img / 255.0, title='all_sift_matches', env=env, biggest_dim=show_biggest_dim)

  match_img = cv2.drawMatches(
    L_img, ref_kp,
    R_img, tgt_kp,
    matches[:len(less_than_100)], R_img.copy(), flags=0)
  imshow(match_img / 255.0, title='less_than_100_dist_sift_matches', env=env, biggest_dim=show_biggest_dim)

  match_img = cv2.drawMatches(
    L_img, ref_kp,
    R_img, tgt_kp,
    matches[:10], R_img.copy(), flags=0)
  imshow(match_img / 255.0, title='top_10_sift_matches', env=env, biggest_dim=show_biggest_dim)

  match_img = cv2.drawMatches(
    L_img, ref_kp,
    R_img, tgt_kp,
    matches[-10:], R_img.copy(), flags=0)
  imshow(match_img / 255.0, title='bottom_10_sift_matches', env=env, biggest_dim=show_biggest_dim)


def get_unknown_intrinsics(im_h, im_w):
  offset_x = im_w / 2
  offset_y = im_h / 2
  intrinsics = np.array([[1, 0, offset_x],
                         [0, 1, offset_y],
                         [0,         0, 1.000000e+00]], dtype='float32')
  return intrinsics

def get_kitti_simulated_intrinsics(im_h, im_w):

  KITTI_DATASET_CAM_F = 7.215377e+02
  KITTI_DATASET_IMG_W = 1242
  KITTI_DATASET_IMG_H = 375
  dataset_f = KITTI_DATASET_CAM_F / KITTI_DATASET_IMG_H * im_h
  offset_x = (im_w - 1.0) / 2
  offset_y = (im_h - 1.0) / 2
  intrinsics = np.array([[dataset_f, 0, offset_x],
                         [0, dataset_f, offset_y],
                         [0,         0, 1.000000e+00]], dtype='float32')
  return intrinsics

def get_simulated_intrinsics(im_h, im_w):
  #TODO: maybe also learn f, which should be more robust with 3d movies
  offset_x = (im_w - 1.0) / 2
  offset_y = (im_h - 1.0) / 2
  #same as OpensFM prior focal length
  f = 0.85*im_w
  intrinsics = np.array([[f, 0, offset_x],
                         [0, f, offset_y],
                         [0,         0, 1.000000e+00]], dtype='float32')
  return intrinsics

def load_json(file_name, replace_nans=False):
  with open(file_name) as handle:
    json_string = handle.read()
    if replace_nans:
      json_string = json_string.replace('-nan', 'NaN').replace('nan', 'NaN')
    parsed_json = json.loads(json_string)
    return parsed_json

def dump_json(json_dict, filename):
  with open(filename, 'w') as fp:
    json.dump(json_dict, fp, indent=4)


try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import json

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)



def np_to_tensor(np_obj):
  return torch.FloatTensor(np_obj)

def np_to_variable(np_obj):
  return Variable(np_to_tensor(np_obj))

def string_similarity(word0, word1):
  return difflib.SequenceMatcher(None, word0, word1).ratio()

def find_closest_string(word, string_list, cutoff=0.6):
  try:
    return difflib.get_close_matches(word, string_list, cutoff=cutoff)[0]
  except:
    return ''

def generate_nice_palette(N_colors):
  palette = sns.color_palette(None, N_colors)
  return np.array(np.array(palette)*255.0, dtype='uint8')

import numpy as np
import colorsys

def generate_nice_palette_colorsys(N_colors):
    hues = np.linspace(0, 1, N_colors, endpoint=False)
    palette = []
    
    for i in range(N_colors):
        hue = hues[i]
        saturation = 0.9 if i % 2 == 0 else 0.6
        value = 0.8 if i % 4 < 2 else 0.6
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        palette.append(rgb)

    # Randomize the palette
    random.shuffle(palette)

    return np.array(np.array(palette) * 255.0, dtype='uint8')



def generate_random_palette(n, seed=-1):
  import random
  if seed != -1:
    previous_state = random.getstate()
    random.seed(seed)
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b))
  if seed != -1:
    random.setstate(previous_state)
  return ret
generate_random_colors = generate_random_palette

def superpixels_image(image, num_segments=50):
  from skimage.segmentation import slic
  from skimage.segmentation import mark_boundaries
  cv2.SuperpixelSEEDS.getLabelContourMask(image)
  segments = slic(image.transpose((1, 2, 0))/50.0, n_segments=num_segments, sigma=5)
  imshow(image, title='image')
  imshow(segments, title='segments')
  return segments

def xrotation_deg_torch(deg, four_dims=False):
  return xrotation_rad_torch(torch_deg2rad(deg), four_dims)

def yrotation_deg_torch(deg, four_dims=False):
  return yrotation_rad_torch(torch_deg2rad(deg), four_dims)

def zrotation_deg_torch(deg, four_dims=False):
  return zrotation_rad_torch(torch_deg2rad(deg), four_dims)

def xrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
                     torch.cat([ones[:, None], zeros[:, None], zeros[:, None]], dim=-1)[:,:,None],
                     torch.cat([zeros[:, None], c[:, None], s[:, None]], dim=-1)[:,:,None],
                     torch.cat([zeros[:, None], -s[:, None], c[:, None]], dim=-1)[:,:,None]
                      ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[ones, zeros, zeros, zeros],
                     [zeros, c, s, zeros],
                     [zeros, -s, c, zeros],
                     [zeros, zeros, zeros, ones]])

def yrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
      torch.cat([c[:, None], zeros[:, None], s[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], ones[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([-s[:, None], zeros[:, None], c[:, None]], dim=-1)[:, :, None]
    ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[c, zeros, s, zeros],
                      [zeros, ones, zeros, zeros],
                      [-s, zeros, c, zeros],
                      [zeros, zeros, zeros, ones]])

def zrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
      torch.cat([c[:, None], -s[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([s[:, None],  c[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], zeros[:, None], ones[:, None]], dim=-1)[:, :, None]
    ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[c, zeros, s, zeros],
                      [zeros, ones, zeros, zeros],
                      [-s, zeros, c, zeros],
                      [zeros, zeros, zeros, ones]])

def xrotation_deg(deg, four_dims=False):
  return xrotation_rad(np.deg2rad(deg), four_dims)

def yrotation_deg(deg, four_dims=False):
  return yrotation_rad(np.deg2rad(deg), four_dims)
def zrotation_deg(deg, four_dims=False):
  return zrotation_rad(np.deg2rad(deg), four_dims)

def xrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
  else:
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])

def yrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
  else:
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def zrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  else:
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def edges_semantic(semantic_instance_map):
  from skimage import feature

  sk_edges = feature.canny(semantic_instance_map, sigma=1) * 1.0
  return sk_edges

def edges_image(img):
  from skimage import feature

  gray = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_BGR2GRAY)
  th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  #edges = cv2.Canny(gray, th / 2, th, )
  sk_edges = feature.canny(gray, sigma=3) * 1.0
  return sk_edges

def linear_pcl_to_abs_pcl(linear_pcl):
  assert len(linear_pcl.shape) == 4
  if not type(linear_pcl) is torch.Tensor:
    raise Exception('Only implemented for tocrch tensor')
  return torch.abs(linear_pcl)

def abs_pcl_to_linear_pcl(abs_pcl):
  assert len(abs_pcl.shape) == 4
  if not type(abs_pcl) is torch.Tensor:
    raise Exception('Only implemented for torch tensor')
  height, width  = abs_pcl.shape[2:]
  mask = torch.FloatTensor(np.ones((1, 3, height, width)))
  mask[:, 0, :, :int(width/2)] = -1
  mask[:, 1, :int(height/2)] = -1
  if abs_pcl.is_cuda:
    mask = mask.cuda()
  pcl = abs_pcl * mask
  return pcl

def linear_pcl_to_log_pcl(linear_pcl, zero_log_bias=0.01):
  assert len(linear_pcl.shape) == 4
  if not type(linear_pcl) is torch.Tensor:
    raise Exception('Only implemented for tocrch tensor')
  return torch.log(torch.abs(linear_pcl) + zero_log_bias)

def log_pcl_to_linear_pcl(log_pcl, zero_log_bias=0.01):
  assert len(log_pcl.shape) == 4
  if not type(log_pcl) is torch.Tensor:
    raise Exception('Only implemented for torch tensor')
  pcl = torch.exp(log_pcl) - zero_log_bias
  height, width  = pcl.shape[2:]
  mask = torch.FloatTensor(np.ones((1, 3, height, width)))
  mask[:, 0, :, :int(width/2)] = -1
  mask[:, 1, :int(height/2)] = -1
  if pcl.is_cuda:
    mask = mask.cuda()
  pcl = pcl * mask
  return pcl


def numpy_batch_mm(A, B):
  #performs B matrix multiplications of tensors B x ? X W times B X W X ?
  return np.matmul(A, B)

def sample_gumbel(shape, eps=1e-20, cuda=False):
  if cuda:
    U = torch.rand(shape).cuda()
  else:
    U = torch.rand(shape)
  return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
  y = logits + sample_gumbel(logits.size(), logits.is_cuda)
  return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
  """
  input: [*, n_class]
  return: [*, n_class] an one-hot vector
  """
  y = gumbel_softmax_sample(logits, temperature)
  shape = y.size()
  _, ind = y.max(dim=-1)
  y_hard = torch.zeros_like(y).view(-1, shape[-1])
  y_hard.scatter_(1, ind.view(-1, 1), 1)
  y_hard = y_hard.view(*shape)
  return (y_hard - y).detach() + y

def load_matlab_mat(filename):
  from scipy.io import loadmat
  return loadmat(filename)


def save_checkpoint_pickles(save_path, others_to_pickle, is_best):
  for (prefix, object) in others_to_pickle.items():
    dump_to_pickle(save_path / ('{}_latest.pckl').format(prefix), object)
  if is_best:
    for prefix in others_to_pickle.keys():
      shutil.copyfile(save_path / ('{}_latest.pckl').format(prefix),
                      save_path / ('{}_best.pckl').format(prefix))

def save_checkpoint(save_path, nets_to_save, is_best, other_objects_to_pickle=None):
  print('Saving checkpoint in: ' + str(save_path))
  save_path = Path(save_path)
  for (prefix, state) in nets_to_save.items():
    torch.save(state, save_path / ('{}_latest.pth.tar').format(prefix))
  if is_best:
    for prefix in nets_to_save.keys():
      shutil.copyfile(save_path / ('{}_latest.pth.tar').format(prefix),
                      save_path / ('{}_best.pth.tar').format(prefix))
  if not other_objects_to_pickle is None:
    save_checkpoint_pickles(save_path, other_objects_to_pickle, is_best)
  print('Saved in: ' + str(save_path))

def reject_outliers(data, m=2):
  return data[abs(data - np.mean(data)) < m * np.std(data)]


def dilate(mask, dilation_percentage=0.01, dilation_pixels=-1):
  if dilation_pixels == -1:
    kernel_size = int(width * dilation_percentage)
  else:
    kernel_size = dilation_pixels
  assert len(mask.shape) == 2
  height, width = mask.shape
  mask_dilated = cv2.dilate(np.array(mask, dtype='uint8'),
                            np.ones((kernel_size, kernel_size)))
  return mask_dilated

def set_optimizer_lr(optimizer, new_lr):
  for group in optimizer.param_groups:
    group['lr'] = new_lr
  return optimizer

def get_optimizer_lr(optimizer):
  return optimizer.state_dict()['param_groups'][0]['lr']

def erode(mask, dilation_percentage=0.01):
  assert len(mask.shape) == 2
  height, width = mask.shape
  mask_eroded = cv2.erode(np.array(mask, dtype='uint8'),
                          np.ones((int(width * dilation_percentage), int(width * dilation_percentage))))
  return mask_eroded

def bb_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
  if interArea == 0:
    return 0
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
  boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)

  # return the intersection over union value
  return iou

class FixSampleDataset:
  def __init__(self, dataset, samples_to_fix = -1, replication_factor=-1):
    self.dataset = dataset
    self.replication_factor = replication_factor
    if type(samples_to_fix) is int and samples_to_fix == -1:
      # just get a random one
      self.fixed_samples = [self.dataset.__getitem__(np.random.randint(0, len(self.dataset)))]
    else:
      if not type(samples_to_fix) is list:
        samples_to_fix = list(samples_to_fix)
      self.fixed_samples = [self.dataset.__getitem__(k) for k in samples_to_fix]

  def __len__(self):
    if self.replication_factor == -1:
      return len(self.fixed_samples)
    else:
      return len(self.fixed_samples) * self.replication_factor

  def set_replication_factor(self, replication_factor):
    self.replication_factor = replication_factor

  def __getattr__(self, item):
    return getattr(self.dataset, item)

  def __getitem__(self, item):
    item = np.random.randint(0, len(self.fixed_samples))
    return self.fixed_samples[item]


def pad_vector(vec, pad, axis=0, value=0):
  """
  args:
      vec - vector to pad
      pad - the size to pad to
      axis - dimension to pad

  return:
      a new tensor padded to 'pad' in dimension 'axis'
  """
  assert type(vec) is np.ndarray, "pad_vector only implemented for numpy array and is {}".format(type(vec))

  pad_size = list(vec.shape)
  pad_size[axis] = pad - vec.shape[axis]
  return np.concatenate([vec, value*np.ones(pad_size, dtype=vec.dtype)], axis=axis)

def get_gpu_stats(counts=10, desired_time_diffs_ms=0):
  gpus = [dict(gpu=0, mem=0) for _ in GPUtil.getGPUs()]
  for _ in range(counts):
    t0 = time.time()
    for gpu_i, gpu in enumerate(GPUtil.getGPUs()):
      gpus[gpu_i]['gpu_usage'] += gpu.load
      gpus[gpu_i]['mem_usage'] += gpu.memoryUsed / gpu.memoryTotal
      time_diff_s = time.time() - t0
      if time_diff_s < desired_time_diffs_ms / 1000.0:
        time.sleep((desired_time_diffs_ms - time_diff_s) / 1000.0)
      t0 = time.time()


  for gpu_i, gpu in enumerate(GPUtil.getGPUs()):
    gpus[gpu_i]['gpu_usage'] /= counts
    gpus[gpu_i]['mem_usage'] /= counts
    gpus[gpu_i]['mem_total'] = gpu.memoryTotal

  return gpus

def get_file_size_bytes(file):
  file_size = os.stat(file).st_size
  return file_size



def get_random_number_from_timestamp():
  return time.time_ns() % 2 ** 32

def print_env_variables():
  for k, v in os.environ.items():
    print("{}: {}".format(k, v))


def checkpoint_can_be_loaded(checkpoint):
  try:
    a = torch.load(checkpoint, map_location=torch.device('cpu'))
  except Exception as e:
    print(e)
    return False
  return True

def delete_all_checkpoints_except_last_loadable(save_folder):
  checkpoints = sorted([save_folder + '/' + k for k in listdir(save_folder, prepend_folder=False) if
                        k.startswith('checkpoint_') and k.endswith('.pth.tar')])

  some_loaded = False
  while len(checkpoints) > 0:
    checkpoint = checkpoints.pop(-1)
    if not some_loaded:
      if checkpoint_can_be_loaded(checkpoint):
        some_loaded = True
      else:
        # it's an invalid checkpoint, so raise exception and delete manually, or check that there's not a mistake with
        # checkpoint_can_be_loaded
        raise Exception("Checkpoint {} could not be loaded".format(checkpoint))
    else:
      os.remove(checkpoint)
  return


def process_in_parallel_or_not(function, elements, parallel, use_pathos=False, num_cpus=-1):
  from pathos.multiprocessing import Pool
  if parallel:
    if use_pathos:
      pool = Pool()
      pool.apply(function, elements)

    if num_cpus > 0:
      return p_map(function, elements, num_cpus=num_cpus)
    else:
      return p_map(function, elements)
  else:
    returns = []
    for k in tqdm(elements):
      returns.append(function(k))

  return returns

def get_directory_and_file(filepath):
  dirname = os.path.dirname(filepath)
  if not os.path.isdir(filepath):
    filename = filepath.split('/')[-1]
  else:
    filename = ''
  return dirname, filename

def print_nvidia_smi():
  nvidia_smi_output = subprocess.run(["nvidia-smi"])
  print(nvidia_smi_output)

def get_current_git_commit():
  repo = git.Repo(search_parent_directories=True)
  sha = repo.head.object.hexsha

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def sort_files_by_number(files):
    # Define a custom sorting function that extracts the first integer from the file path
    def sort_key(file_path):
        match = re.search(r'\d+', file_path.split('/')[-1])
        return int(match.group()) if match else 0

    return sorted(files, key=sort_key)

def add_frame_counter(frames, color=(255,0,0)):
    assert type(frames) is list and all(frame.shape[0] == 3 and frame.dtype == np.uint8 for frame in frames)
    
    for i, frame in enumerate(frames):
        frame = frame.transpose(1, 2, 0)  # Convert to shape (height, width, 3)
        H, W, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
        frame = cv2.putText(np.ascontiguousarray(frame), text, (W // 10, H // 10), font, 1, color, 2)
        frames[i] = frame.transpose(2, 0, 1)  # Convert back to shape (3, height, width)
    return frames
        
  
if __name__ == '__main__':
  images = np.random.uniform(0, 1, size=(50, 3, 128, 128))
  imshow(images, title='random_video', verbosity=1)

  imshow(tile_images(images, border_pixels=4), title='tiles_example')

  # print_nvidia_smi()
  random.randint(1,3)
  # gpus = get_gpu_stats(counts=10, desired_time_diffs_ms=0)
  # print(gpus)
  # a = 1

