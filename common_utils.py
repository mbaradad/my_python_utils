#force 3.5 div to make it compatible with both 2.7 and 3.5
from __future__ import division
#TODO: move imports to functions, to make loading faster
from _curses import raw
#import open3d as o3d

try:
  import cPickle as pickle
except:
  import _pickle as pickle
import os
import cv2

import glob
import shutil
import time

import seaborn as sns
import warnings
import random
import argparse
import matplotlib
from my_python_utils.visdom_visualizations import *

from imageio import imwrite
from scipy import misc
import struct

from scipy.ndimage.filters import gaussian_filter

import imageio
import torch
import numpy as np
from PIL import Image, ImageDraw

import datetime
import json
import difflib

from plyfile import PlyData, PlyElement

import torch.nn.functional as F
from torch.autograd import Variable

from scipy.linalg import lstsq
import socket

global VISDOM_BIGGEST_DIM
VISDOM_BIGGEST_DIM = 300

def get_hostname():
  return socket.gethostname()

def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
  return gpus

def gettimedatestring():
  return datetime.datetime.now().strftime("%m-%d-%H:%M")

from multiprocess import Lock
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

def read_text_file_lines(filename):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      lines.append(line.replace('\n',''))
  return lines

def write_text_file_lines(lines, file):
  with open(file, 'w') as file_handler:
    for item in lines:
      file_handler.write("%s\n" % item)

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

def is_headless_execution():
  #if there is a display, we are running locally
  return 'DISPLAY' in os.environ.keys()
if not is_headless_execution():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')


def chunk_list(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def png_16_bits_imread(file):
  return cv2.imread(file, -cv2.IMREAD_ANYDEPTH)

def cv2_imread(file, return_BGR=False):
  im = cv2.imread(file).transpose(2,0,1)
  if return_BGR:
    return im
  return im[::-1, :, :]

def load_image_tile(filename, top, bottom, left, right, dtype='uint8'):
  #img = pyvips.Image.new_from_file(filename, access='sequential')
  roi = img.crop(left, top, right - left, bottom - top)
  mem_img = roi.write_to_memory()

  # Make a numpy array from that buffer object
  nparr = np.ndarray(buffer=mem_img, dtype=dtype,
                     shape=[roi.height, roi.width, roi.bands])
  return nparr

def cv2_imwrite(im, file, rgb=True, normalize=False):
  if len(im.shape) == 3 and im.shape[0] == 3:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min())/(im.max() - im.min())
    im = np.array(255.0*im, dtype='uint8')
  imwrite(file, im)

def merge_side_by_side(im1, im2):
  assert im1.shape == im2.shape
  im_canvas = np.concatenate((np.zeros_like(im1), np.zeros_like(im1)), axis=2)
  im_canvas[:,:,:im1.shape[-1]] = im1
  im_canvas[:,:,im1.shape[-1]:] = im2
  return im_canvas

def visdom_histogram(array, win=None, title=None, env='PYCHARM_RUN', vis=None):
  if vis is None:
    vis = global_vis
  opt = dict()
  if not title is None:
    opt['title'] = title
  else:
    opt['title'] = str(win)
  if win is None:
    win = title
  vis.histogram(array, env=env, win=win, opts=opt)


def visdom_barplot(array, env='PYCHARM_RUN', win=None, title=None, vis=None):
  if vis is None:
    vis = global_vis
  opt = dict()
  if not title is None:
    opt['title'] = title
  else:
    opt['title'] = str(win)
  if win is None:
    win = title
  vis.bar(array, env=env, win=win, opts=opt)

def visdom_bar_plot(array, rownames=None, env='PYCHARM_RUN', win=None, title=None, vis=None):
  if vis is None:
    vis = global_vis
  opt = dict()
  if not title is None:
    opt['title'] = title
  else:
    opt['title'] = str(win)
  if win is None:
    win = title
  if not rownames is None:
    opt['rownames'] = rownames
  vis.bar(array, env=env, win=win, opts=opt)


def visdom_boxplot(array, env='main', win='test', title=None, vis=None):
  if vis is None:
    vis = global_vis
  opt = dict()
  if not title is None:
    opt['title'] = title
  else:
    opt['title'] = str(win)
  vis.boxplot(array, env=env, win=win, opts=opt)

def touch(fname, times=None):
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

def tile_images(imgs, tiles, tile_size):
  final_img = np.zeros((3, tiles[0]*tile_size[0], tiles[1]*tile_size[1]))
  n_imgs = len(imgs)
  k = 0
  for i in range(tiles[0]):
    for j in range(tiles[1]):
      tile = myimresize(imgs[k], tile_size)
      final_img[:, i*tile_size[0]:(i+1)*tile_size[0],j*tile_size[1]:(j+1)*tile_size[1]] = tile
      k = k + 1
      if k >= n_imgs:
        break
    if k >= n_imgs:
      break
  return final_img

def str2intlist(v):
 if len(v) == 0:
   return []
 return [int(k) for k in v.split(',')]


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_2d_bbox(im, min_corner_xy, max_corner_xy, color):
  im = Image.fromarray(im.transpose())
  draw = ImageDraw.Draw(im)

  draw.line((min_corner_xy[0], min_corner_xy[1], min_corner_xy[0], max_corner_xy[1]), fill=color)
  draw.line((min_corner_xy[0], min_corner_xy[1], max_corner_xy[0], min_corner_xy[1]), fill=color)
  draw.line((max_corner_xy[0], min_corner_xy[1], max_corner_xy[0], max_corner_xy[1]), fill=color)
  draw.line((min_corner_xy[0], max_corner_xy[1], max_corner_xy[0], max_corner_xy[1]), fill=color)

  return np.array(im).transpose()

def add_line(im, origin_x_y, end_x_y, color=(255, 0, 0)):
  im = Image.fromarray(im.transpose())
  draw = ImageDraw.Draw(im)

  draw.line((origin_x_y[1], origin_x_y[0], end_x_y[1], end_x_y[0]), fill=color)

  return np.array(im).transpose()


def add_circle(im, centers_x_y, radius=5, color=(255, 0, 0)):
  im = Image.fromarray(im.transpose())
  draw = ImageDraw.Draw(im)
  if not type(centers_x_y) is list:
    centers_x_y = [centers_x_y]
  for i in range(len(centers_x_y)):
    center_x_y = centers_x_y[i]
    if type(color) is list:
      actual_color = tuple(color[i])
    else:
      actual_color = color
    draw.ellipse((center_x_y[1] - radius, center_x_y[0] - radius, center_x_y[1] + radius, center_x_y[0] + radius), fill=actual_color)

  return np.array(im).transpose()

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

def imshow(im, title='none', path=None, biggest_dim=None, normalize_image=True, max_batch_display=10, window=None, env='PYCHARM_RUN', fps=None, vis=None, add_ranges=False, return_image=False):
  if 'VISDOM_BIGGEST_DIM' in globals() and biggest_dim is None:
    global VISDOM_BIGGEST_DIM
    biggest_dim = VISDOM_BIGGEST_DIM
  if window is None:
    window = title
  if type(im) == 'string':
    #it is a path
    pic = Image.open(im)
    im = np.array(pic, dtype='float32')
  im = tonumpy(im)
  postfix = ''
  if im.dtype == np.bool:
    im = im*1.0
  if add_ranges:
    postfix = '_max_{:.2f}_min_{:.2f}'.format(im.max(), im.min())
  if im.dtype == 'uint8':
    im = im / 255.0
  if len(im.shape) > 4:
    raise Exception('Im has more than 4 dims')
  if len(im.shape) == 4 and im.shape[0] == 1:
   im = im[0,:,:,:]
  if len(im.shape) == 3 and im.shape[-1] in [1,3]:
    #put automatically channel first if its last
    im = im.transpose((2,0,1))
  if len(im.shape) == 2:
    #expand first if 1 channel image
    im = im[None,:,:]
  if not biggest_dim is None and len(im.shape) == 3:
    im = scale_image_biggest_dim(im, biggest_dim)
  if normalize_image and im.max() != im.min():
    im = (im - im.min())/(im.max() - im.min())
  if path is None:
    if window is None:
      window = title
    if len(im.shape) == 4:
      vidshow_vis(im, title=title, window=window, env=env, vis=vis, biggest_dim=biggest_dim, fps=fps)
    else:
      imshow_vis(im, title=title + postfix, window=window, env=env, vis=vis)
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

def count_trainable_parameters(network):
  n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
  return n_parameters

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


default_side_colors = np.array(sns.color_palette("hls", 12)) * 255.0
default_corner_colors = (np.array(sns.color_palette("hls", 8)) * 255.0).transpose()

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

def get_grads(network):
  params = list(network.parameters())
  grads = [p.grad for p in params]
  return grads

def scale_img_to_fit_canvas(img, canvas_height, canvas_width):
  im = Image.fromarray(img.transpose((1,2,0)))
  im.thumbnail((canvas_height, canvas_width), Image.ANTIALIAS)
  return np.array(im).transpose((2,0,1))


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

def show_pointcloud_errors(coords, errors, title='none', win=None, env='PYCHARM_RUN', markersize=3, max_points=10000,
                    force_aspect_ratio=True, valid_mask=None, nice_plot_rotation=3):
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

def show_pointcloud(coords, np_colors=None, title='none', win=None, env='PYCHARM_RUN', markersize=3, max_points=10000,
                    force_aspect_ratio=True, valid_mask=None, nice_plot_rotation=3, labels=None):
  if np_colors is None:
    np_colors = np.ones(coords.shape)
  coords = tonumpy(coords)
  np_colors = tonumpy(np_colors)
  if np_colors.dtype == 'float32':
    if np_colors.max() > 1.0:
      np_colors = np.array(np_colors, dtype='uint8')
    else:
      np_colors = np.array(np_colors * 255.0, dtype='uint8')
  if type(coords) is list:
    for k in range(len(np_colors)):
      if np_colors[k] is None:
        np_colors[k] = np.ones(coords[k].shape)
    np_colors = np.concatenate(np_colors, axis=0)
    coords = np.concatenate(coords, axis=0)

  assert coords.shape == np_colors.shape
  if len(coords.shape) == 3:
    coords = coords.reshape((3, -1))
  if len(np_colors.shape) == 3:
    np_colors = np_colors.reshape((3, -1))
  assert len(coords.shape) == 2
  if coords.shape[0] == 3:
    coords = coords.transpose()
    np_colors = np_colors.transpose()

  if not valid_mask is None:
    valid_mask = np.array(valid_mask, dtype='bool')
    coords = coords[valid_mask]
    np_colors = np_colors[valid_mask]

  if max_points != -1 and coords.shape[0] > max_points:
    selected_positions = random.sample(range(coords.shape[0]), max_points)
    coords = coords[selected_positions]
    np_colors = np_colors[selected_positions]
    if not labels is None:
      labels = [labels[k] for k in selected_positions]
  if win is None:
    win = title
  if force_aspect_ratio:
    #move this to use plotly

    #add coords on a bounding box, to force
    min_coords = coords.min(0)
    max_coords = coords.max(0)

    bbox_coords = generate_bbox_coords(min_coords, max_coords)
    bbox_colors = np.array([(255,255,255)]*8)
    bbox_coords = np.array(bbox_coords)
    coords = np.concatenate((coords, bbox_coords), axis=0)
    np_colors = np.concatenate((np_colors, bbox_colors), axis=0)
    if not labels is None:
      labels.extend(['' for _ in range(8)])
  if nice_plot_rotation == 1:
    plot_coords = np.matmul(xrotation_rad(np.deg2rad(90)), coords.transpose()).transpose()
    plot_coords = np.matmul(yrotation_rad(np.deg2rad(180)), plot_coords.transpose()).transpose()
    plot_coords = np.matmul(zrotation_rad(np.deg2rad(-45)), plot_coords.transpose()).transpose()
  elif nice_plot_rotation == 2:
    plot_coords = np.matmul(xrotation_rad(np.deg2rad(-90)), coords.transpose()).transpose()
    plot_coords = np.matmul(yrotation_rad(np.deg2rad(180)), plot_coords.transpose()).transpose()
    plot_coords = np.matmul(zrotation_rad(np.deg2rad(-45 + 180)), plot_coords.transpose()).transpose()
  elif nice_plot_rotation == 3:
    plot_coords = np.matmul(xrotation_rad(np.deg2rad(-90)), coords.transpose()).transpose()
    plot_coords = np.matmul(yrotation_rad(np.deg2rad(180)), plot_coords.transpose()).transpose()
    plot_coords = np.matmul(zrotation_rad(np.deg2rad(-45 )), plot_coords.transpose()).transpose()
  elif nice_plot_rotation == 4:
    plot_coords = np.matmul(xrotation_rad(np.deg2rad(-90)), coords.transpose()).transpose()
    plot_coords = np.matmul(yrotation_rad(np.deg2rad(180)), plot_coords.transpose()).transpose()
    plot_coords = np.matmul(zrotation_rad(np.deg2rad(180 )), plot_coords.transpose()).transpose()
  else:
    plot_coords = coords

  from visdom import _markerColorCheck
  # we need to construct our own colors to override marker plotly options
  # and allow custom hover (to show real coords)
  colors = _markerColorCheck(np_colors, plot_coords, np.ones(len(plot_coords), dtype='uint8'), 1)
  hovertext = ['x:{:.2f}\ny:{:.2f}\nz:{:.2f}\n'.format(float(k[0]), float(k[1]), float(k[2])) for k in coords]
  if not labels is None:
    assert len(labels) == len(hovertext)
    hovertext = [hovertext[k] + ' {}'.format(labels[k]) for k in range(len(hovertext))]

  global_vis.scatter(plot_coords, env=env, win=win,
              opts={'webgl': True,
                    'title': title,
                    'name': 'scatter',
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
                          'size': markersize,
                          'symbol': 'dot',
                          'color': colors[1],
                          'line': {
                              'color': '#000000',
                              'width': 0,
                          }
                        }
                      }}}
                    })
  '''
  verts = list()
  polygons = list()
  size = 0.2
  N = 100
  for k in range(len(coords[:N])):
    actual_coords = coords[k]
    verts.append(actual_coords - np.array((size, 0,0)))
    verts.append(actual_coords - np.array((-size, 0,0)))
    verts.append(actual_coords - np.array((0, size, 0)))
    verts.append(actual_coords - np.array((0, -size, 0)))
    polygons.append([k*3, k*3 + 1, k*3 + 2])
  viz.mesh(X=np.array(verts), Y=np.array(polygons), opts={'opacity':0.5, 'color': colors[:N]}, win='test')
  '''

def list_dir(folder, prepend_folder):
  if prepend_folder:
    return [folder + '/' + k for k in os.listdir(folder)]
  return os.listdir(folder)

def print_float(number):
  return "{:.2f}".format(number)

def imshow_matplotlib(im, path):
  imwrite(path,np.transpose(im, (1, 2, 0)))

import matplotlib.pyplot as pyplt
def histogram_image(array, nbins=20, legend=None):
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

  image = misc.imread(tmp_fig_file)
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


def fit_plane_np(data_points):
  assert data_points.shape[0] == 3
  A = np.c_[data_points[0, :], data_points[1, :], np.ones(data_points.shape[1])]
  C, _, _, _ = lstsq(A, data_points[2, :])
  # The new z will be the z where the original directions intersect the plane C
  p_n_0 = np.array((C[0], C[1], -1, C[2]))
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

  image = misc.imread(tmp_fig_file)

  return image

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

def mkdir(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)
  return

def delete(file):
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

def cv2_resize(image, target_shape, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_shape[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_shape[::-1], interpolation=interpolation).transpose((2, 0, 1))

def best_centercrop_image(image, height, width, return_rescaled_size=False):
  image_height, image_width = image.shape[-2:]
  im_crop_height_shape = (int(height), int(image_width * height / image_height))
  im_crop_width_shape = (int(image_height * width / image_width), int(width))
  # if we crop on the height dimension, there must be enough pixels on the width
  if im_crop_height_shape[1] >= width:
    rescaled_size = im_crop_height_shape
  else:
    # crop over width
    rescaled_size = im_crop_width_shape
  resized_image = cv2_resize(image, rescaled_size)
  center_cropped = crop_center(resized_image, (height, width))
  if return_rescaled_size:
    return center_cropped, rescaled_size
  else:
    return center_cropped


def undo_img_normalization(img, dataset='movies'):
  if img.shape[0] == 1:
    return img
  from sintel_data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
  std = MovieSequenceDataset.getstd()
  mean = MovieSequenceDataset.getmean()
  if not type(img) is np.ndarray:
    std = torch.FloatTensor(std)
    mean = torch.FloatTensor(mean)
  if dataset != 'movies':
    raise Exception('Not implemented!')
  if img.shape[0] == 3:
    img = img*std[:,None,None] + mean[:,None,None]
  else:
    img = img * std + mean
  return img

def do_img_normalization(img, dataset='movies'):
  from sintel_data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
  if dataset != 'movies':
    raise Exception('Not implemented!')
  if img.shape[0] == 3:
    return (img - np.reshape(MovieSequenceDataset.getmean(),(3,1,1))) / np.reshape(MovieSequenceDataset.getstd(),(3,1,1))
  else:
    return (img - MovieSequenceDataset.getmean()) / MovieSequenceDataset.getstd()

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

def tonumpy(tensor):
  if type(tensor) is np.ndarray:
    return tensor
  if tensor.requires_grad:
    tensor = tensor.detach()
  if type(tensor) is torch.autograd.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.detach().numpy()

def totorch(numpy_array):
  return torch.FloatTensor(numpy_array)

def tovariable(array):
  if type(array) == np.ndarray:
    array = totorch(array)
  return Variable(array)

def extrinsic_mat_to_pose(mat):
  return 1

def pose_to_extrinsic(mat):
  return 1

def subset_frames(get_dataset=False, fps=4):
  from sintel_data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
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


url = ("http://maps.googleapis.com/maps/api/geocode/json?"
       "address=googleplex&sensor=false")
print(get_jsonparsed_data(url))

def np_to_tensor(np_obj):
  return torch.FloatTensor(np_obj)

def np_to_variable(np_obj):
  return Variable(np_to_tensor(np_obj))

def find_closest_string(word, string_list):
  return difflib.get_close_matches(word, string_list)[0]

def generate_palette(n, seed=-1):
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

def linear_plc_to_abs_plc(linear_plc):
  assert len(linear_plc.shape) == 4
  if not type(linear_plc) is torch.Tensor:
    raise Exception('Only implemented for tocrch tensor')
  return torch.abs(linear_plc)

def abs_plc_to_linear_plc(abs_plc):
  assert len(abs_plc.shape) == 4
  if not type(abs_plc) is torch.Tensor:
    raise Exception('Only implemented for torch tensor')
  height, width  = abs_plc.shape[2:]
  mask = torch.FloatTensor(np.ones((1, 3, height, width)))
  mask[:, 0, :, :int(width/2)] = -1
  mask[:, 1, :int(height/2)] = -1
  if abs_plc.is_cuda:
    mask = mask.cuda()
  plc = abs_plc*mask
  return plc

def linear_plc_to_log_plc(linear_plc, zero_log_bias):
  assert len(linear_plc.shape) == 4
  if not type(linear_plc) is torch.Tensor:
    raise Exception('Only implemented for tocrch tensor')
  return torch.log(torch.abs(linear_plc) + zero_log_bias)

def log_plc_to_linear_plc(log_plc, zero_log_bias):
  assert len(log_plc.shape) == 4
  if not type(log_plc) is torch.Tensor:
    raise Exception('Only implemented for torch tensor')
  plc = torch.exp(log_plc) - zero_log_bias
  height, width  = plc.shape[2:]
  mask = torch.FloatTensor(np.ones((1, 3, height, width)))
  mask[:, 0, :, :int(width/2)] = -1
  mask[:, 1, :int(height/2)] = -1
  if plc.is_cuda:
    mask = mask.cuda()
  plc = plc*mask
  return plc


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
  for (prefix, state) in nets_to_save.items():
    torch.save(state, save_path / ('{}_model_latest.pth.tar').format(prefix))
  if is_best:
    for prefix in nets_to_save.keys():
      shutil.copyfile(save_path / ('{}_model_latest.pth.tar').format(prefix),
                      save_path / ('{}_model_best.pth.tar').format(prefix))
  if not other_objects_to_pickle is None:
    save_checkpoint_pickles(save_path, other_objects_to_pickle, is_best)
  print('Saved in: ' + save_path)

def reject_outliers(data, m=2):
  return data[abs(data - np.mean(data)) < m * np.std(data)]


if __name__ == '__main__':
  test_image = 'imgs_to_test/img_9294.jpg'
  img = scale_img_to_fit_canvas(cv2_imread(test_image), 720, 720)

  detect_lines(img)

  import time
  then = time.time()
  for k in range(100):
    img = load_image_tile(panoramic_image, 0, 1024, 0, 2048)
  print("full loading time: {}".format(time.time() - then))
  then = time.time()
  for k in range(100):
    tile = load_image_tile(panoramic_image, 500, 600, 500, 600)
  print("Loading tile time: {}".format(time.time() - then))
  imshow(tile, biggest_dim=600, title='tile')
  imshow(img, biggest_dim=600, title='img')
  while True:
    a = 1