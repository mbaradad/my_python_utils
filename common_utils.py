#force 3.5 div to make it compatible with both 2.7 and 3.5
from __future__ import division
#TODO: move imports to functions, to make loading faster
try:
  import cPickle as pickle
except:
  import _pickle as cPickle
import os

import argparse
import matplotlib
if not 'NO_VISDOM' in os.environ.keys():
  import visdom
  global_vis = visdom.Visdom(port=12890, server='http://vision02')

import png
import numpy as np

from imageio import imwrite
from scipy import misc
from skimage.transform import resize

import imageio
import torch
from torch.autograd import Variable
import cv2
import numpy as np

from skvideo.io import FFmpegWriter as VideoWriter
import tempfile
from PIL import Image
import datetime
import json
import difflib
from skimage import feature

from plyfile import PlyData, PlyElement

def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = range(len(gpus_arg.split(',')))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  return gpus

def gettimedatestring():
  return datetime.datetime.now().strftime("%m-%d-%H:%M")

def read_text_file_lines(filename):
  lines = list()
  with open(filename) as f:
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
  matplotlib.use('Agg')


def png_16_bits_imread(file):
  return cv2.imread(file, -cv2.IMREAD_ANYDEPTH)

def cv2_imread(file, return_BGR=False):
  im = cv2.imread(file).transpose(2,0,1)
  if return_BGR:
    return im
  return im[::-1, :, :]

def scale_image_biggest_dim(im, biggest_dim):
  #if it is a video, resize inside the video
  if im.shape[1] > im.shape[2]:
    scale = im.shape[1] / (biggest_dim + 0.0)
  else:
    scale = im.shape[2] / (biggest_dim + 0.0)
  target_imshape = (int(im.shape[1]/scale), int(im.shape[2]/scale))
  if im.shape[0] == 1:
    im = myimresize(im[0], target_shape=(target_imshape))[None,:,:]
  else:
    im = myimresize(im, target_shape=target_imshape)
  return im

def visdom_histogram(array, env, win, title=None, vis=None):
  if vis is None:
    vis = global_vis
  opt = dict()
  if not title is None:
    opt['title'] = title
  vis.histogram(array, env=env, win=win, opts=opt)

def touch(fname, times=None):
  with open(fname, 'a'):
    os.utime(fname, times)

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

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def imshow(im, title='none', path=None, biggest_dim=None, normalize_image=True, max_batch_display=10, window=None, env='main', fps=None, vis=None, add_ranges=False):
  if title is None:
    raise Exception("Imshow error: Title can't be empty!")
  if window is None:
    window = title
  im = tonumpy(im)
  postfix = ''
  if type(im) == 'string':
    #it is a path
    pic = Image.open(im)
    im = np.array(pic, dtype='float32')
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

def imshow_vis(im, title=None, window=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.win_exists(title)
  vis.image(im, win=window, opts=opts, env=env)

def visdom_dict(dict_to_plot, title=None, window=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  properties = []
  for k in dict_to_plot.keys():
    properties.append({'type': 'text', 'name': str(k), 'value': str(dict_to_plot[k])})
  vis.properties(properties, win=window, opts=opts, env=env)

def vidshow_vis(video, title=None, window=None, env=None, vis=None, biggest_dim=None, fps=10):
  if vis is None:
    vis = global_vis
  if video.shape[1] == 1 or video.shape[1] == 3:
    video = video.transpose(0,2,3,1)
  if video.shape[-1] == 1:
    #if one channel, replicate it
    video = np.tile(video,(1,1,1,3))
  opts = dict()
  if not title is None:
    opts['caption'] = title
    opts['fps'] = fps
  if not video.dtype is np.uint8:
    video = np.array(video * 255, dtype='uint8')
  vis.win_exists(title)
  if window is None:
    window = title

  videofile = '/tmp/%s.ogv' % next(tempfile._get_candidate_names())
  writer = VideoWriter(videofile, inputdict={'-r': str(fps)})
  for i in range(video.shape[0]):
    if biggest_dim is None:
      actual_frame = video[i]
    else:
      actual_frame = np.transpose(scale_image_biggest_dim(np.transpose(video[i]), biggest_dim))
    writer.writeFrame(actual_frame)
  writer.close()
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

def create_plane_pointcloud_coords(center, normal, extent, samples, color=(255,255,255)):
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

def show_pointcloud(coords, colors=None, title='none', win=None, env=None, markersize=5, subsample=-1, force_aspect_ratio=True, valid_mask=None):
  if colors is None:
    colors = np.ones(coords.shape)
  if type(coords) is list:
    for k in range(len(colors)):
      if colors[k] is None:
        colors[k] = np.ones(coords[k].shape)
    colors = np.concatenate(colors, axis=0)
    coords = np.concatenate(coords, axis=0)
  if not valid_mask is None:
    coords = coords[valid_mask]
    colors = colors[valid_mask]
  if subsample != -1:
    coords = coords[::subsample]
    colors = colors[::subsample]
  if win is None:
    win = title
  if force_aspect_ratio:
    #add coords on a bounding box, to force
    max_coord = np.abs(coords).max()
    bbox_coords = list()
    for to_bits in range(8):
      i = int(to_bits / 4)
      j = int(to_bits / 2)
      k = int(to_bits % 2)
      bbox_coords.append((max_coord*(-1)**i, max_coord*(-1)**j, max_coord*(-1)**k))
    bbox_colors = np.array([(255,255,255)]*8)
    bbox_coords = np.array(bbox_coords)
    coords = np.concatenate((coords, bbox_coords), axis=0)
    colors = np.concatenate((colors, bbox_colors), axis=0)

  global_vis.scatter(coords, env=env, win=win,
              opts={'markercolor':colors,
                    'markersize' : markersize,
                    'webgl': False,
                    'markersymbol': 'dot',
                    'title':title,
                    'name': 'scatter'})

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


def fov_to_intrinsic(fov_x, fov_y, width, height):
  fx = width / (2 * np.tan(fov_x / 2))
  fy = height / (2 * np.tan(fov_y / 2))
  intrinsics = np.array(((fx, 0, width / 2),
                       (0, fy, height / 2),
                       (0,  0,         1)))
  return intrinsics, np.linalg.inv(intrinsics)

def dump_pointcloud(coords, colors, file_name, valid_mask=None):
  if not valid_mask is None:
    coords = coords[valid_mask]
    colors = colors[valid_mask]
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
  plydata.write(file_name + '.ply')

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

def myimresize(img, target_shape):
  max = img.max(); min = img.min()
  if max > min:
    img = (img - min)/(max - min)
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    img = np.transpose(resize(np.transpose(img, (1,2,0)), target_shape, mode='constant'), (2,0,1))
  else:
    img = resize(img, target_shape, mode='constant')
  if max > min:
    return (img*(max - min) + min)
  else:
    return img

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


def undo_img_normalization(img, dataset='movies'):
  if img.shape[0] == 1:
    return img
  from data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
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
  from data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
  if dataset != 'movies':
    raise Exception('Not implemented!')
  if img.shape[0] == 3:
    return (img - np.reshape(MovieSequenceDataset.getmean(),(3,1,1))) / np.reshape(MovieSequenceDataset.getstd(),(3,1,1))
  else:
    return (img - MovieSequenceDataset.getmean()) / MovieSequenceDataset.getstd()

def tonumpy(tensor):
  if type(tensor) is np.ndarray:
    return tensor
  if type(tensor) is torch.autograd.Variable:
    if tensor.requires_grad:
      tensor = tensor.detach()
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
  from data.datagenerators.movie_sequence_dataset import MovieSequenceDataset
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

def get_sift_matches(gray_ref_img, gray_tgt_img, mask_ref_and_target=None, dist_threshold=-1, N_MATCHES=-1):
  sift = cv2.xfeatures2d.SIFT_create()
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
    less_than_100 = [d for d in sim_distances if d < 100]

    match_img = cv2.drawMatches(
      gray_L_img, ref_kp,
      gray_R_img, tgt_kp,
      matches, gray_R_img.copy(), flags=0)
    imshow(match_img / 255.0, title='all_sift_matches', env=env, biggest_dim=1000)

    match_img = cv2.drawMatches(
      gray_L_img, ref_kp,
      gray_R_img, tgt_kp,
      matches[:len(less_than_100)], gray_R_img.copy(), flags=0)
    imshow(match_img / 255.0, title='less_than_100_dist_sift_matches', env=env, biggest_dim=1000)


    match_img = cv2.drawMatches(
      gray_L_img, ref_kp,
      gray_R_img, tgt_kp,
      matches[:10], gray_R_img.copy(), flags=0)
    imshow(match_img / 255.0, title='top_10_sift_matches', env=env, biggest_dim=1000)

    match_img = cv2.drawMatches(
      gray_L_img, ref_kp,
      gray_R_img, tgt_kp,
      matches[-10:], gray_R_img.copy(), flags=0)
    imshow(match_img / 255.0, title='bottom_10_sift_matches', env=env, biggest_dim=1000)

  return L_pts, R_pts, sim_distances

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

def load_json(file_name):
  with open(file_name) as handle:
    return json.loads(handle.read())
def dump_json(json_dict, filename):
  with open(filename, 'w') as fp:
    json.dump(json_dict, fp, indent=4)

def np_to_tensor(np_obj):
  return torch.FloatTensor(np_obj)

def np_to_variable(np_obj):
  return Variable(np_to_tensor(np_obj))

def find_closest_string(word, string_list):
  return difflib.get_close_matches(word, string_list)[0]

def superpixels_image(image, num_segments=50):
  from skimage.segmentation import slic
  from skimage.segmentation import mark_boundaries
  cv2.SuperpixelSEEDS.getLabelContourMask(image)
  segments = slic(image.transpose((1, 2, 0))/50.0, n_segments=num_segments, sigma=5)
  imshow(image, title='image')
  imshow(segments, title='segments')
  return segments


def edges_semantic(semantic_instance_map):
  sk_edges = feature.canny(semantic_instance_map, sigma=1) * 1.0
  return sk_edges

def edges_image(img):
  gray = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_BGR2GRAY)
  th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  #edges = cv2.Canny(gray, th / 2, th, )
  sk_edges = feature.canny(gray, sigma=3) * 1.0
  return sk_edges

  '''
  model = compute_edgelets(img)
  vis_edgelets(img, model)
  rectified = rectify_image(img, 4, algorithm='independent')
  imshow(rectified, biggest_dim=400, env='test_lines', title='img')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  minLineLength = 10
  maxLineGap = 10
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
  for x1, y1, x2, y2 in lines[0]:
      cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  imshow(img[:,:,::-1], biggest_dim=400, env='test_lines', title='img_lines_canny')
  '''



if __name__ == '__main__':
  #dataset = subset_frames(get_dataset=True, width=240)
  #for i in range(len(dataset)):
  #  ref_img, tgt_imgs, intrinsics, intrinsics_inv, filename, base_index = dataset.__getitem__(i)#range(len(dataset))[-20])
  #  tgt_img = tgt_imgs[1]
  #  compute_sift_image(ref_img, tgt_img, mask=None, make_plots=True)
  from scipy.io import loadmat
  colors = loadmat('data/color150.mat')['colors']
  import csv

  with open('data/object150_info.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    object_names = [k[-1].split(';')[0] for k in list(reader)]
  create_legend_classes(object_names, colors, range(150))
