import subprocess
import time
from datetime import datetime
import numpy as np

start_alpha = 0.1
start_beta = 0.1
start_gamma = 0.1
temperature = 10
epoch = 5

processes = []
for alpha in np.arange(start_alpha, 1.1, 0.1):
    cmd = [
        'conda', 'run', '-n', 'tf_od', 
        'python', 'val_distillation.py',
        '--start_alpha', str(alpha),
        '--start_beta', str(start_beta),
        '--start_gamma', str(start_gamma),
        '--start_temperature', str(temperature),
        '--epoch', str(epoch)
    ]
    processes.append(subprocess.Popen(cmd))
    time.sleep(30)

for process in processes:
    process.wait()