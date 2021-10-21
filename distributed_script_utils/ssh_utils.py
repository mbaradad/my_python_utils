from my_python_utils.common_utils import *
from p_tqdm import p_map
import paramiko

config_script="""
export HOME=/data/vision/torralba/movies_sfm/home
cd $HOME

#add my_libs
export PATH="$HOME/programs/bin:$PATH"
# added by Anaconda2 installer
if [[ $PATH = *$HOME/anaconda* ]]
then
    echo "Anaconda path already present, wont be added again!"
else
    export PATH="$HOME/anaconda3/bin:$PATH"
fi

source activate default_env37
source $HOME/.bash_commands/all
"""

get_gpu_stats_script=config_script + \
"""

cd $HOME/no_training_cnn;
export PYTHONPATH=.
python my_python_utils/distributed_script_utils/get_nvidia_stats.py

"""

antonio_hosts = ['visiongpu03',
                 'visiongpu06',
                 'visiongpu07',
                 'visiongpu08',
                 'visiongpu09',
                 'visiongpu10',
                 'visiongpu11',
                 'visiongpu15',
                 'visiongpu17',
                 'visiongpu19',
                 'visiongpu22',
                 'visiongpu24',
                 'visiongpu26',
                 'visiongpu34',
                 'visiongpu35',
                 'visiongpu36',
                 'visiongpu37',
                 'visiongpu38',
                 'visiongpu39',
                 'visiongpu49']

all_hosts = ["visiongpu{}".format(str(k).zfill(2)) for k in range(3, 55)]
port = 22
username = "mbaradad"
if not 'SSH_PASSWORD' in os.environ.keys():
    print("Write password!")
    password = input("Enter your ssh password: ")
else:
    password = os.environ["SSH_PASSWORD"]

def run_script_on_machines(get_gpu_stats_script, hosts, parallel=True, print_output=True, process_output_func=None):
  def single_host_task(host, command):
      ssh = paramiko.SSHClient()
      ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      try:
          ssh.connect(host, port, username, password, timeout=10)

          stdin, stdout, stderr = ssh.exec_command(command)
          linesout = stdout.readlines()
          lineserr = stderr.readlines()

          if not process_output_func is None:
            output = process_output_func(linesout)
          else:
            output = linesout
          return (host, output)
      except Exception as e:
          #if 'Authentication failed' in str(e):
          #  print("Authentication failed. Check that the password is correct (user: {}; password: {})".format(username, password))
          return (host, "Could not be reached. Exception: " + str(e))

  command_outputs = p_map(lambda x: single_host_task(x,get_gpu_stats_script), hosts, num_cpus=10 if parallel else 1)
  if print_output:
    for machine, outputs in command_outputs:
        print('{}: {}'.format(machine, outputs))
  return command_outputs

if __name__ == "__main__":
  def process_output_func(linesout):
    return json.loads(linesout[0])
  # run_script_on_machines(get_gpu_stats_script, all_hosts, process_output_func)

  get_running_process_script = config_script + \
    """
    ps aux | grep train | grep _bw | grep mbaradad | grep -v grep
    """

  kill_process_script = config_script + \
    """
    pkill train
    pkill python
    """

  excluded_machines = ['visiongpu09', 'visiongpu37']
  hosts = [k for k in all_hosts if not k in excluded_machines]

  run_script_on_machines(kill_process_script, all_hosts, parallel=True)
  run_script_on_machines(get_running_process_script, all_hosts, parallel=True)