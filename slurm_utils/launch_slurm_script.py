import argparse
import time
import subprocess

import os

def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')

parser = argparse.ArgumentParser(description='Launch slurm script')

parser.add_argument('script', type=str, help='scipt to launch')
parser.add_argument('--relaunch-if-dead', default='True', type=str2bool, help='path to dataset')
parser.add_argument('--min-run-time-seconds', default=5, type=str2bool, help='min time of running ')
parser.add_argument('--max-total-time-days', default=30, type=int, help='whether to test if the datasets have the expected number of files')
parser.add_argument('--max-runs', default=100, type=int, help='whether to test if the datasets have the expected number of files')
parser.add_argument('--init-slurm-id', default=-1, type=int, help='id if the slurm script is already running. This will wait for the slurm with id to finish, and then execute the sript')


def not_completed(slurm_id):
  output = subprocess.run('sacct --format="User,JobID,JobName%30,Elapsed,State,NodeList" | grep -v TIMEOUT | grep mbaradad', shell=True, capture_output=True)
  lines = str(output.stdout).split('\\n')
  for line in lines:
    if str(slurm_id) in line:
      if 'CANCEL' in line or 'COMPLET' in line:
        return False

  return True

if __name__ == '__main__':
  args = parser.parse_args()

  if not os.path.exists(args.script):
    print("Script {} does not exist!".format(args.script))

  t_init = time.time()
  t_1 = time.time()

  n = 0

  while t_1 - t_init < args.max_total_time_days * 3600 * 24:
    print("Running slurm script {} for time: {}".format(args.script, n + 1))
    t_0 = time.time()

    if args.init_slurm_id == -1:
      output = subprocess.run("sbatch --qos=sched_level_2 " + args.script, shell=True, capture_output=True)
      slurm_id = int(str(output.stdout).split(' job ')[-1].split('\\')[0])

    print("Currently running slurm job with id: {}".format(slurm_id))
    while not_completed(slurm_id):
      time.sleep(5)

    t_1 = time.time()
    run_time = t_1 - t_0
    if run_time < args.min_run_time_seconds:
      print("The slurm script only run for {} seconds, less than {} required.".format(run_time,  args.min_run_time_seconds))
      exit(0)

    n += 1

  print("Finished running. The slurm script executed {} times in {} days".format(n, (t_1 - t_init) / 3600 / 24))