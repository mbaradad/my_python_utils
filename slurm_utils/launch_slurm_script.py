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
parser.add_argument('--max-runs', default=200, type=int, help='whether to test if the datasets have the expected number of files')
parser.add_argument('--slurm-id', default=-1, type=int, help='id if the slurm script is already running. This will wait for the slurm with id to finish, and then execute the sript')
parser.add_argument('--partition-8', action='store_true', help='whether to use partition 8 or the rest')
parser.add_argument('--qos', type=int, choices=[0,1,2], default=2, help='whether to use partition 8 or the rest')

def check_status(slurm_id, was_running):
  output = subprocess.run('sacct --format="User,JobID,JobName%30,Elapsed,State,NodeList" | grep -v TIMEOUT | grep mbaradad', shell=True, capture_output=True)
  lines = str(output.stdout).split('\\n')
  is_running = False
  found = False
  for line in lines:
    if str(slurm_id) in line:
      found = True
      if 'CANCEL' in line or 'COMPLET' in line or 'NODE_FAIL' in line:
        return False, is_running
      if 'RUN' in line:
        is_running = True

  # if previously found running and not anymore,
  if was_running and not found:
    return False, False

  # return not_completed, is_running
  return True, is_running

if __name__ == '__main__':
  args = parser.parse_args()

  if not os.path.exists(args.script):
    print("Script {} does not exist!".format(args.script))

  t_init = time.time()
  t_1 = time.time()

  n = 0

  while t_1 - t_init < args.max_total_time_days * 3600 * 24 and n < args.max_runs:
    print("Running slurm script {} for time: {}".format(args.script, n + 1))
    t_0 = time.time()
    from datetime import datetime

    now = datetime.now()  # current date and time

    if args.slurm_id == -1:
      commands = ["sbatch"]
      if args.partition_8:
        commands.append("--partition=sched_system_all_8")
      #commands.append("--exclude=node0019")
      if args.qos != 0:
        commands.append("--qos=sched_level_{}".format(args.qos))
      commands.append(args.script)
      print(" ".join(commands))

      output = subprocess.run(commands, shell=False, capture_output=True)
      date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
      print("Slurm job launched at:", date_time)
      try:
        slurm_id_str = str(output.stdout).split(' job ')[-1]
        slurm_id = int(slurm_id_str.split('\\')[0])
      except:
        print("Exception while getting slurmid, this happened in the past when launching from a compute node instead of the login node")
        slurm_id = -1
    else:
      slurm_id = args.slurm_id

    print("Currently running slurm job with id: {}".format(slurm_id))
    not_completed, was_running = check_status(slurm_id, False)
    while not_completed:
      not_completed, was_running = check_status(slurm_id, was_running)
      time.sleep(5)

    t_1 = time.time()
    run_time = t_1 - t_0
    if run_time < args.min_run_time_seconds:
      print("The slurm script only run for {} seconds, less than {} required.".format(run_time,  args.min_run_time_seconds))
      exit(0)

    n += 1

  print("Finished running. The slurm script executed {} times in {} days".format(n, (t_1 - t_init) / 3600 / 24))