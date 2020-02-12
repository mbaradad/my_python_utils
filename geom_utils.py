import numpy as np
from math import sqrt
import torch

'''
  From https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
  
  Input: expects 3xN matrix of points
  Returns R,t
  R = 3x3 rotation matrix
  t = 3x1 column vector
'''

def rigid_transform_3D(A, B):
  assert len(A) == len(B)
  if type(A) is np.ndarray:
    A = np.mat(A)
  if type(B) is np.ndarray:
    B = np.mat(B)
  A = A.T
  B = B.T

  num_rows, num_cols = A.shape;

  if num_rows != 3:
    raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

  [num_rows, num_cols] = B.shape;
  if num_rows != 3:
    raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

  # find mean column wise
  centroid_A = np.mean(A, axis=1)
  centroid_B = np.mean(B, axis=1)

  # subtract mean
  Am = A - np.tile(centroid_A, (1, num_cols))
  Bm = B - np.tile(centroid_B, (1, num_cols))

  # dot is matrix multiplication for array
  H = Am * np.transpose(Bm)

  # find rotation
  U, S, Vt = np.linalg.svd(H)
  R = Vt.T * U.T

  # special reflection case
  if np.linalg.det(R) < 0:
    #print("det(R) < R, reflection detected!, correcting for it ...\n");
    Vt[2, :] *= -1
    R = Vt.T * U.T

  t = -R * centroid_A + centroid_B

  return np.array(R), np.array(t)




import sys
from my_python_utils.common_utils import *

def compute_normals_from_closest_image_coords(coords, mask=None):
    #TODO: maybe compute normals with bigger region?
    x_coords = coords[:,0,:,:]
    y_coords = coords[:,1,:,:]
    z_coords = coords[:,2,:,:]

    if type(coords) is torch.Tensor or type(coords) is torch.nn.parameter.Parameter:
      ts = torch.cat((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None,:-1,1:]), dim=1)
      ls = torch.cat((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), dim=1)
      cs = torch.cat((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None,1:,1:]), dim=1)

      n = torch.cross((ls - cs),(ts - cs), dim=1)
      n_norm = n/(torch.sqrt(torch.abs((n*n).sum(1) + 1e-20))[:,None,:,:])
    else:
      ts = np.concatenate((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None,:-1,1:]), axis=1)
      ls = np.concatenate((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), axis=1)
      cs = np.concatenate((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None,1:,1:]), axis=1)

      n = np.cross((ls - cs),(ts - cs), axis=1)
      n_norm = n/(np.sqrt(np.abs((n*n).sum(1) + 1e-20))[:,None,:,:])

    if not mask is None:
      assert len(mask.shape) == 4
      valid_ts = mask[:,:, :-1, 1:]
      valid_ls = mask[:,:, 1:, :-1]
      valid_cs = mask[:,:, 1:, 1:]
      if type(mask) is torch.tensor:
        final_mask = valid_ts * valid_ls * valid_cs
      else:
        final_mask = np.logical_and(np.logical_and(valid_ts, valid_ls), valid_cs)
      return n_norm, final_mask
    else:
      return n_norm

def compute_normals_from_pcl(coords, viewpoint, radius=0.2, max_nn=30):
  assert len(viewpoint.shape) == 1
  assert len(coords.shape) == 2
  assert type(coords) == np.ndarray
  assert coords.shape[0] == 3
  pcd = o3d.geometry.PointCloud()
  coords_to_pcd = coords.transpose()
  pcd.points = o3d.utility.Vector3dVector(coords_to_pcd)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=max_nn))
  estimated_normals = np.array(pcd.normals)
  # orient towards viewpoint
  normal_viewpoint_signs = np.sign((estimated_normals*(coords_to_pcd - viewpoint[None,:])).sum(-1))
  corrected_normals = (estimated_normals*normal_viewpoint_signs[:,None]).transpose().reshape(coords.shape)
  return corrected_normals

def compute_plane_intersections_from_camera(extrinsics, K, P, mask):
  assert len(P.shape) == 4 and P.shape[0] == 4
  extrinsics

from kornia import pixel2cam as k_pixel2cam
from kornia import create_meshgrid, convert_points_to_homogeneous

pixel_coords_cpu = None
pixel_coords_cuda = None

def get_id_grid(height, width):
  return create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2

def set_id_grid(height, width, to_cuda=False):
  global pixel_coords_cpu, pixel_coords_cuda

  pixel_grid = get_id_grid(height, width)
  if to_cuda:
    pixel_coords_cuda = pixel_grid.cuda()
  else:
    pixel_coords_cpu = pixel_grid.cpu()

def get_id_grid(height, width):
  grid = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2
  return convert_points_to_homogeneous(grid)


def make_4x4_K(K):
  batch_size = K.shape[0]
  zeros = torch.zeros((batch_size,3,1))
  with_1 = torch.Tensor(np.array((0,0,0,1)))[None,:None:].expand(batch_size,1,4)
  if K.is_cuda:
    zeros = zeros.cuda()
    with_1 = with_1.cuda()
  K = torch.cat((K, zeros), axis=2)
  K = torch.cat((K, with_1), axis=1)

  return K

def pixel2cam(depth, K):
  global pixel_coords_cpu, pixel_coords_cuda
  if len(depth.shape) == 4:
    assert depth.shape[1] == 1
    depth = depth[1]
  assert len(depth.shape) == 3
  assert K.shape[1] == K.shape[2]
  assert depth.shape[0] == K.shape[0]

  K = make_4x4_K(K)
  intrinsics_inv = torch.inverse(K)

  height, width = depth.shape[-2:]
  if depth.is_cuda:
    # to avoid recomputing the id_grid if it is not necessary
    if (pixel_coords_cuda is None) or pixel_coords_cuda.size(2) != height or pixel_coords_cuda.size(3) != width:
      set_id_grid(height, width, to_cuda=True)
    pixel_coords = pixel_coords_cuda
  else:
    if (pixel_coords_cpu is None) or pixel_coords_cpu.size(2) != height or pixel_coords_cpu.size(3) != width:
      set_id_grid(height, width, to_cuda=False)
    pixel_coords = pixel_coords_cpu

  batch_size = depth.shape[0]
  pcl = k_pixel2cam(depth[:,None,:,:], intrinsics_inv, pixel_coords.expand(batch_size, -1, -1, -1))
  return pcl.permute(0,3,1,2)

def compute_plane_intersections(pose, intrinsics, plane, mask):
  from my_python_utils.common_utils import show_pointcloud
  camera_center = pose[:3, 3]
  cam_dirs = pixel2cam(torch.ones((1,*mask.shape)), intrinsics[None,:,:])[0]
  cam_dirs = (pose[:3, :3] @ cam_dirs.reshape((3, -1))).reshape(cam_dirs.shape)
  camera_ends = camera_center[:,None,None] + cam_dirs

  camera_center = torch.cat((camera_center, torch.ones(1)))
  camera_ends = torch.cat((camera_ends, torch.ones(1, *camera_ends.shape[1:])))

  plucker_mats = construct_plucker_matrices(camera_center[None, :], camera_ends.reshape((4,-1)).transpose(0,1))
  plane_intersections = plucker_mats @ plane[None,:,None]
  plane_intersections = plane_intersections[:, :3, 0] / plane_intersections[:, 3]

  plane_intersections = plane_intersections.reshape((*camera_ends.shape[1:], 3)).permute((2,0,1))
  masked_plane_intersections = torch.zeros((3, *mask.shape))
  masked_plane_intersections[:, mask.bool()] = plane_intersections[:, mask.bool()]

  return masked_plane_intersections

def construct_plucker_matrices(points_a, points_b):
  assert len(points_a.shape) == 2
  assert len(points_b.shape) == 2
  assert points_a.shape[1] == 4
  assert points_b.shape[1] == 4

  matrices = points_a[:,None,:] * points_b[:,:,None] - points_a[:,:,None] * points_b[:,None,:]

  return matrices