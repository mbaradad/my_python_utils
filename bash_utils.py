from shutil import rmtree
from p_tqdm import p_map
import os
import glob

# useful to delete folders with lots of subfolders/files
def parallel_delete(foldername, max_level_to_parallelize, workers=50):
  # finds folders recursively up to n levels
  def single_delete(folder):
    try:
      rmtree(folder)
    except:
      pass

  for actual_level in range(max_level_to_parallelize, -1, -1):
    print('Listing at level: {}'.format(actual_level))
    level_files = glob.glob(foldername + '/' + '*/' * actual_level)
    #level_dirs = filter(os.path.isdir, level_filter)
    print('{} files found at level: {}'.format(len(level_files), actual_level))
    print('Starting parallel delete at level: {}'.format(actual_level))
    p_map(single_delete, level_files, num_cpus = workers)
    print('End parallel delete at level: {}'.format(actual_level))

if __name__ == "__main__":
  parallel_delete('/data/vision/torralba/scratch/mbaradad/big_brother/colmap_reconstructions/s01', max_level_to_parallelize=1)
