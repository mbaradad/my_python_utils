import numpy as np
from math import sqrt

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


