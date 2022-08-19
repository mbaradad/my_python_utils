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

# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def transform_unit_A_unit_B(a, b):
  raise Exception("TODO!!")
  is_numpy = type(a) is np.ndarray
  if is_numpy:
    a = torch.Tensor(a)
    b = torch.Tensor(b)
  assert len(a.shape) == 2
  assert a.shape == b.shape
  assert a.shape[0] == 3 and a.shape[1] == 1
  v = torch.cross(a, b)
  s = torch.norm(v)
  c = (a * b).sum(0)

  R = torch.eye(3) + cross_product_mat_sub_x_torch(v) + cross_product_mat_sub_x_torch(v)**2 * (1 - c) / s**2

  if is_numpy:
    return R.numpy()

  return R


def rigid_transform_3D_batched(A, B, mask):
  # returns the rigid transform (R, t) that such that R @ A + t gives minimal distance to B, assuming one to one mapping from A to B.
  # A and B are torch arrays of size batch_size x 3 x N, mask indicates which of the N points should be used for A and B
  assert A.shape == B.shape
  assert len(A.shape) == 3, "Should be BX x 3 x N"
  assert A.shape[-2] == 3, "Should be BX x 3 x N"
  assert len(mask.shape) == 2
  assert mask.shape[0] == A.shape[0] and mask.shape[1] == A.shape[2]
  mask = mask[:,None,:]

  assert not A.requires_grad and not B.requires_grad, "This was not implemented to be backpropagated, though may not require many changes"

  N = A.shape[-1]

  if mask is None:
    mask_sum = torch.ones((A.shape[0], A.shape[1])).to(A.device)
  else:
    mask_sum = mask.sum(-1)

  # find mean column wise
  centroid_A = (A * mask).sum(-1)/ mask_sum
  centroid_B = (B * mask).sum(-1)/ mask_sum

  # subtract mean
  Am = A - torch.tile(centroid_A[:,:,None], (1, 1, N))
  Bm = B - torch.tile(centroid_B[:,:,None], (1, 1, N))

  # put zeros on masked regions so that it does not penalize
  Am = Am * mask
  Bm = Bm * mask

  # dot is matrix multiplication for array
  H = Am @ Bm.transpose(1,2)

  # find rotation
  U, S, Vt = torch.linalg.svd(H)
  R = Vt.transpose(1,2) @ U.transpose(1,2)

  # special reflection case
  #print("det(R) < R, reflection detected!, correcting for it ...\n");
  Vt_inverted = Vt.clone()
  Vt_inverted[:, 2, :] *= -1
  R_inverted = Vt_inverted.transpose(1,2) @ U.transpose(1,2)

  negative_det = torch.linalg.det(R) < 0
  R = R * (1 - negative_det[:,None,None] * 1.0) + R_inverted * negative_det[:,None,None]

  t = (-R @ centroid_A[:,:,None])[:,:,0] + centroid_B

  return R.float(), t.float()

def rigid_transform_3D_np(A, B):
  assert len(A) == len(B)
  if type(A) is np.ndarray:
    A = np.mat(A)
  if type(B) is np.ndarray:
    B = np.mat(B)

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
    # TODO: maybe change cs to be [..., :-1,:-1] as it seems more intuitive
    assert coords.shape[1] == 3 and len(coords.shape) == 4
    assert mask is None or (mask.shape[0] == coords.shape[0] and mask.shape[2:] == coords.shape[2:])

    x_coords = coords[:,0,:,:]
    y_coords = coords[:,1,:,:]
    z_coords = coords[:,2,:,:]

    if type(coords) is torch.Tensor or type(coords) is torch.nn.parameter.Parameter:
      ts = torch.cat((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None,:-1,1:]), dim=1)
      ls = torch.cat((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), dim=1)
      cs = torch.cat((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None,1:,1:]), dim=1)

      n = torch.cross((ls - cs),(ts - cs), dim=1) * 1e10

      # if normals appear incorrect, it may be becuase of the 1e-100, if the scale of the pcl is two small 1e-5,
      # it was giving errors with a constant of 1e-20 (and it was replaced to 1e-100), we do this to avoid nans
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
      if type(mask) is torch.Tensor:
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
  estimated_normals = np.array(pcd.geometry_to_fit)
  # orient towards viewpoint
  normal_viewpoint_signs = np.sign((estimated_normals*(coords_to_pcd - viewpoint[None,:])).sum(-1))
  corrected_normals = (estimated_normals*normal_viewpoint_signs[:,None]).transpose().reshape(coords.shape)
  return corrected_normals

def compute_plane_intersections_from_camera(extrinsics, K, P, mask):
  assert len(P.shape) == 4 and P.shape[0] == 4
  extrinsics

if int(torch.__version__.split('.')[0]) > 0:
  try:
    from kornia.geometry.camera.pinhole import pixel2cam as k_pixel2cam
    from kornia.utils import create_meshgrid
    from kornia.geometry.conversions import convert_points_to_homogeneous
  except Exception as e:
    print("Failed to import kornia. Shouldn't be an issue if no geom utils are used")



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

def euler2mat_torch(angle):
  """Convert euler angles to rotation matrix.
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
  """
  B = angle.size(0)
  x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

  cosz = torch.cos(z)
  sinz = torch.sin(z)

  zeros = z.detach() * 0
  ones = zeros.detach() + 1
  zmat = torch.stack([cosz, -sinz, zeros,
                      sinz, cosz, zeros,
                      zeros, zeros, ones], dim=1).reshape(B, 3, 3)

  cosy = torch.cos(y)
  siny = torch.sin(y)

  ymat = torch.stack([cosy, zeros, siny,
                      zeros, ones, zeros,
                      -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

  cosx = torch.cos(x)
  sinx = torch.sin(x)

  xmat = torch.stack([ones, zeros, zeros,
                      zeros, cosx, -sinx,
                      zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

  rotMat = xmat @ ymat @ zmat
  return rotMat

def rotation_matrix_two_vectors(vector_from, vector_to):
  # returns r st np.matmul(r, a) = b
  v = np.cross(vector_from, vector_to)
  c = np.dot(vector_from, vector_to)
  s = np.linalg.norm(v)
  I = np.identity(3)
  vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
  k = np.matrix(vXStr)
  r = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
  return np.array(r)



if __name__ == '__main__':
  N = 30
  A = np.random.normal(0,1,(3, N))
  R = zrotation_deg(50) @ yrotation_deg(50) @ xrotation_deg(60)
  t = np.random.normal(0,1,3)
  B = R @ A + t[:,None]

  '''
  show_pointcloud([A.transpose(),B.transpose()],[np.array([(255,0,0)]*N),np.array([(0,255,0)]*N)], title='before_aligning')

  R_np, t_np = rigid_transform_3D_np(A, B)
  A_after_np_aligning = (R_np @ A + t_np)
  show_pointcloud([A_after_np_aligning.transpose(),B.transpose()],[np.array([(255,0,0)]*N),np.array([(0,255,0)]*N)], title='after_np_aligning')

  '''

  R_torch, t_torch = rigid_transform_3D_batched(totorch(A)[None], totorch(B)[None], mask=np.ones((1, N)))
  A_after_torch_aligning = (tonumpy(R_torch) @ A + tonumpy(t_torch)[:,:,None])[0]
  show_pointcloud([A_after_torch_aligning.transpose(),B.transpose()],[np.array([(255,0,0)]*N),np.array([(0,255,0)]*N)], title='after_torch_aligning')


  BS = 10
  N_valid = N // 2
  multi_A = torch.tile(totorch(A)[None], (BS,1,1))
  multi_B = torch.tile(totorch(B)[None], (BS,1,1))
  padded_mask = np.ones((BS, N), dtype='uint8')
  padded_mask[:,N_valid:] = 0
  R_torch, t_torch = rigid_transform_3D_batched(multi_A, multi_B, mask=padded_mask)

  multi_A_after_torch_aligning = (R_torch @ multi_A + t_torch[:,:,None])
  for k in range(0, BS, BS//2):
    show_pointcloud([multi_A_after_torch_aligning[k,:,:N_valid].transpose(0,1),multi_B[k,:,:N_valid].transpose(0,1)],
                    [np.array([(255,0,0)] * N_valid),np.array([(0,255,0)] * N_valid)],
                    title='after_multi_torch_aligning_{}'.format(k))