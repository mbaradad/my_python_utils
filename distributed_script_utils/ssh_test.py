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

def single_host_task(host, command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, port, username, password, timeout=10)
    except:
        return ''
    stdin, stdout, stderr = ssh.exec_command(command)
    linesout = stdout.readlines()
    lineserr = stderr.readlines()

    return (host, json.loads(linesout[0]))


nvidia_outputs = p_map(lambda x: single_host_task(x,get_gpu_stats_script), all_hosts)
min_memory = 0
available_gpu_per_machine = dict()
for machine, stats in nvidia_outputs:
    exit(0)

experiment_script=
