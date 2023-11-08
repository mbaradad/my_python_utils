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
                 'visiongpu49',
                 'visiongpu58',
                 'visiongpu59',
                 'visiongpu60',
                 'visiongpu61']

all_hosts = ["visiongpu{}".format(str(k).zfill(2)) for k in range(3, 62)]
port = 22
username = "mbaradad"
if not 'SSH_PASSWORD' in os.environ.keys():
    print("Write password!")
    password = input("Enter your ssh password: ")
else:
    password = os.environ["SSH_PASSWORD"]

def run_script_on_machines(get_gpu_stats_script, hosts, parallel=True, print_output=True, process_output_func=None, debug=False):
  def single_host_task(host, command):
      if debug:
        print("Connecting to host: " + host)
      ssh = paramiko.SSHClient()
      ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      try:
          if debug:
            print("ssh to host: " + host + "")
          ssh.connect(host, port, username, password, timeout=2)

          stdin, stdout, stderr = ssh.exec_command(command)
          linesout = stdout.readlines()
          lineserr = stderr.readlines()

          if not process_output_func is None:
            output = process_output_func(linesout)
          else:
            output = linesout
          if debug:
            print("Finished connection to host: " + host)
          return (host, output)
      except Exception as e:
          #if 'Authentication failed' in str(e):
          #  print("Authentication failed. Check that the password is correct (user: {}; password: {})".format(username, password))
          return (host, "Could not be reached. Exception: " + str(e))

  command_outputs = process_in_parallel_or_not(lambda x: single_host_task(x,get_gpu_stats_script), hosts, parallel)
  if print_output:
    for machine, outputs in command_outputs:
        if len(outputs) == 0:
          outputs = ['no output']
        if not type(outputs) is list:
          outputs = [outputs]
        for output in outputs:
          while len(output) > 0 and output[-1] == '\n':
            output = output[:-1]
          print(machine + ': ' + output)

  return command_outputs


if __name__ == "__main__":
  def process_output_func(linesout):
    return json.loads(linesout[0])
  # run_script_on_machines(get_gpu_stats_script, all_hosts, process_output_func)

  get_running_process_script = config_script + \
    """
    ps aux | grep mbaradad | grep controlnet | grep -v grep
    """

  kill_process_script = config_script + \
    """
    pkill train
    pkill python
    """

  # excluded_machines = ['visiongpu09', 'visiongpu37']
  excluded_machines = ['visiongpu44']
  if not excluded_machines is None:
    hosts = [k for k in all_hosts if not k in excluded_machines]

  print("If it gets stuck, run with parallel=False, see what machine is making it get stuck and put it in excluded_machines!")
  parallel = True
  debug = False
  #run_script_on_machines(kill_process_script, all_hosts, parallel=True)
  run_script_on_machines(get_running_process_script, all_hosts, parallel=parallel, debug=debug, print_output=True)

  a = 1