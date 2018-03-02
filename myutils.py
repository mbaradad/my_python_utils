import cPickle as pickle
import os
import shutil

import matplotlib
import visdom
from imageio import imwrite
from scipy import misc
from skimage.transform import resize

import imageio
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from skvideo.io import FFmpegWriter as VideoWriter
import tempfile
from PIL import Image

def select_gpus(gpus_arg):
  #so that default gpu is one of the selected, instead of 0
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = range(len(gpus_arg.split(',')))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  return gpus

def read_text_file_lines(filename):
  lines = list()
  with open(filename) as f:
    for line in f:
      lines.append(line.replace('\n',''))
  return lines

def dump_str_list(examples_dirs, examples_list_file):
  with open(examples_list_file, 'w') as file_handler:
    for item in examples_dirs:
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



def is_local_execution():
  #if there is a display, we are running locally
  return 'DISPLAY' in os.environ.keys()
if not is_local_execution():
  matplotlib.use('Agg')

from PIL import Image

vis = visdom.Visdom(port=12890)

def imshow(im, path=None, biggest_dim=None, normalize_image=True, max_batch_display=10, title=None, window=None, env=None, fps=None):
  im = tonumpy(im)
  if type(im) == 'string':
    #it is a path
    pic = Image.open(im)
    im = np.array(pic, dtype='float32')
  if im.dtype == 'uint8':
    im = im / 255.0
  if len(im.shape) > 4:
    raise Exception('Im has more than 4 dims')
  if len(im.shape) == 3 and im.shape[-1] in [1,3]:
    #put automatically channel first if its last
    im = im.transpose((2,0,1))
  if len(im.shape) == 2:
    #expand first if 1 channel image
    im = im[None,:,:]
  if not biggest_dim is None:
    if im.shape[1] > im.shape[2]:
      scale = im.shape[1] / (biggest_dim + 0.0)
    else:
      scale = im.shape[2] / (biggest_dim + 0.0)
    target_imshape = (int(im.shape[1]/scale), int(im.shape[2]/scale))
    if im.shape[0] == 1:
      im = myimresize(im[0], target_shape=(target_imshape))[None,:,:]
    else:
      im = myimresize(im, target_shape=target_imshape)
  if normalize_image and im.max() != im.min():
    im = (im - im.min())/(im.max() - im.min())
  if path is None:
    if len(im.shape) == 4:
      vidshow_vis(im, title=title, window=window, env=env)
    else:
      imshow_vis(im, title=title, window=window, env=env)
  else:
    if len(im.shape) == 4:
      make_gif(im, path=path, fps=fps)
    else:
      imshow_matplotlib(im, path)

def make_gif(ims, path, fps=None):
  if ims.dtype != 'uint8':
    ims = np.array(ims*255, dtype='uint8')
  if ims.shape[1] in [1,3]:
    ims = ims.transpose((0,2,3,1))
  if ims.shape[-1] == 1:
    ims = np.tile(ims, (1,1,1,3))
  with imageio.get_writer(path) as gif_writer:
    for k in range(ims.shape[0]):
      #imsave(ims[k].mean()
      gif_writer.append_data(ims[k])
  if not fps is None:
    gif = imageio.mimread(path)
    imageio.mimsave(path, gif, fps=fps)


def imshow_vis(im, title=None, window=None, env=None):
  opts = dict()
  if not title is None:
    opts['caption'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.win_exists(title)
  if window is None:
    window = title
  vis.image(im, win=window, opts=opts, env=env)

def vidshow_vis(video, title=None, window=None, env=None):
  if video.shape[1] == 1 or video.shape[1] == 3:
    video = video.transpose(0,2,3,1)
  if video.shape[1] == 1:
    video = np.tile(video,(1,1,1,3))
  opts = dict()
  if not title is None:
    opts['caption'] = title
  if not video.dtype is np.uint8:
    video = np.array(video * 255, dtype='uint8')
  vis.win_exists(title)
  if window is None:
    window = title

  videofile = '/tmp/%s.ogv' % next(tempfile._get_candidate_names())
  writer = VideoWriter(videofile)
  for i in range(video.shape[0]):
    im = Image.fromarray(np.transpose(video[i],(2,0,1)))
    # performs a rescale, that fits inside the (720,720)
    im.thumbnail((300, 300), Image.ANTIALIAS)
    writer.writeFrame(np.array(im))
  writer.close()
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

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

  return np.transpose(image,[2,0,1])

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
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    return np.transpose(resize(np.transpose(img, (1,2,0)), target_shape, mode='constant'), (2,0,1))
  else:
    return resize(img, target_shape, mode='constant')

def torch_load(torch_path, gpus):
  if len(gpus) == 0:
    return torch_load_on_cpu(torch_path)
  else:
    return torch.load(torch_path)

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

def undo_img_normalization(img, dataset='movies'):
  from data.movie_sequence_dataset import MovieSequenceDataset
  if dataset != 'movies':
    raise Exception('Not implemented!')
  if img.shape[0] == 3:
    return img*np.reshape(MovieSequenceDataset.getstd(),(3,1,1)) + np.reshape(MovieSequenceDataset.getmean(),(3,1,1))
  else:
    return img * MovieSequenceDataset.getstd() + MovieSequenceDataset.getmean()

def do_img_normalization(img, dataset='movies'):
  from data.movie_sequence_dataset import MovieSequenceDataset
  if dataset != 'movies':
    raise Exception('Not implemented!')
  if img.shape[0] == 3:
    return (img - np.reshape(MovieSequenceDataset.getmean(),(3,1,1))) / np.reshape(MovieSequenceDataset.getstd(),(3,1,1))
  else:
    return (img - MovieSequenceDataset.getmean()) / MovieSequenceDataset.getstd()

def tonumpy(tensor):
  if type(tensor) is np.ndarray:
    return tensor
  if type(tensor) is torch.autograd.variable.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.numpy()

def subset_frames(get_dataset=False, fps=4):
  from data.movie_sequence_dataset import MovieSequenceDataset
  selected_movies = ['pulp_fiction_1994']
  selected_frames =[]
  #refered as indexes at 4 fps
  #selected_frames_4fps.extend(range(400, 500))
  #selected_frames_4fps.extend(range(2204, 2284))
  #selected_frames_4fps.extend(range(2666, 2811))
  #selected_frames_4fps.extend(range(2939, 3000))

  selected_frames.extend(range(6*400,  6*500))
  selected_frames.extend(range(6*2204, 6*2284))
  selected_frames.extend(range(6*2666, 6*2811))
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

def get_sift_matches(gray_ref_img, gray_tgt_img, mask_ref_and_target=None, N_MATCHES=100):
  sift = cv2.xfeatures2d.SIFT_create()
  ref_kp, ref_desc = sift.detectAndCompute(gray_ref_img, None)
  tgt_kp, tgt_desc = sift.detectAndCompute(gray_tgt_img, None)

  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  matches = bf.match(ref_desc, tgt_desc)
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

def compute_sift_image(ref_img, tgt_img, intrinsics, mask_ref_and_target=None, make_plots=True, N_MATCHES=100, env=None):
  if env is None:
    env = 'sift' + '_masked' if  not mask_ref_and_target is None else ''
  if ref_img.shape[0] in [1,3]:
    ref_img = np.transpose(ref_img, (1,2,0))
  if tgt_img.shape[0] in [1, 3]:
    tgt_img = np.transpose(tgt_img, (1,2,0))
  if len(ref_img.shape) == 3 and ref_img.shape[-1] == 3:
    gray_ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
  else:
    gray_ref_img = ref_img
    gray_tgt_img = tgt_img
  if len(ref_img.shape) == 2:
    ref_img = ref_img[:,:,None]
    tgt_img = tgt_img[:, :, None]

  src_pts, dst_pts, ref_kp, tgt_kp, matches = get_sift_matches(gray_ref_img, gray_tgt_img, mask_ref_and_target=None, N_MATCHES=N_MATCHES)
  match_img = cv2.drawMatches(
    gray_ref_img, ref_kp,
    gray_tgt_img, tgt_kp,
    matches, gray_tgt_img.copy(), flags=0)

  E, R, t = get_essential_matrix(src_pts, dst_pts, intrinsics)

  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  matchesMask = mask.ravel().tolist()

  h, w = ref_img.shape[:2]
  pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
  dst = cv2.perspectiveTransform(pts, M)
  #warped_image = cv2.warpPerspective(tgt_img, np.linalg.inv(M), (tgt_img.shape[1], tgt_img.shape[0]))

  tgt_img_with_homo = tgt_img.copy()
  tgt_img_with_homo[:,:,0] = cv2.polylines(tgt_img[:,:,0], [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

  draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                     singlePointColor=None,
                     matchesMask=matchesMask,  # draw only inliers
                     flags=2)

  img3 = cv2.drawMatches(ref_img, ref_kp, tgt_img, tgt_kp, matches, None, **draw_params)

  if make_plots:
    if not mask_ref_and_target is None:
      imshow(mask_ref_and_target[0], title='ref_img_mask', window='ref_img_mask', env=env)
      imshow(mask_ref_and_target[1], title='tgt_img_mask', window='tgt_img_mask', env=env)

    #imshow(warped_image/255.0, title='warped_target', window='warped_target', env=env)
    #print t
    imshow(ref_img / 255.0, title='ref_img', window='ref_img', env=env)
    imshow(cv2.drawKeypoints(gray_ref_img, ref_kp, ref_img.copy()) / 255.0, title='sift_ref_features',
           window='sift_ref_features', env=env)
    imshow(cv2.drawKeypoints(gray_tgt_img, tgt_kp, tgt_img.copy()) / 255.0, title='sift_tgt_features',
         window='sift_tgt_features', env=env)
    imshow(img3 / 255.0, title='transformed', window='transformed', env=env)
    imshow(match_img / 255.0, title='sift_matches', window='sift_matches', env=env)

  print t[-1]
  return E, R, t, src_pts, dst_pts


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

def np_to_tensor(np_obj):
  return torch.FloatTensor(np_obj)

def np_to_variable(np_obj):
  return Variable(np_to_tensor(np_obj))

if __name__ == '__main__':
  dataset = subset_frames(get_dataset=True, width=240)
  for i in range(len(dataset)):
    ref_img, tgt_imgs, intrinsics, intrinsics_inv, filename, base_index = dataset.__getitem__(i)#range(len(dataset))[-20])
    tgt_img = tgt_imgs[1]
    compute_sift_image(ref_img, tgt_img, mask=None, make_plots=True)

