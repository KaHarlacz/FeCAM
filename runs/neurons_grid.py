import itertools
import subprocess
import json
import datetime

epochs = 1000      
learning_rates = [0.000005]
layers_neurons = [
  [256, 256, 128, 64, 16],
  [256, 128, 64, 32, 16],
  [512, 256, 128, 64, 32, 16],
  [256, 256, 128, 128, 32, 32, 16]
]
grid_name = 'ae_neurons_mse_grid'

config_path = './exps/FeCAM_cifar100.json'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_tf_log_dir = f"tf_logs/{grid_name}/" + current_time

for (lr, layers) in itertools.product(learning_rates, layers_neurons):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['epochs'] = epochs
    data['lr'] = lr
    data['maha_alpha'] = 0
    data['maha_beta'] = 1
    data['layers_neurons'] = layers
    data['tf_dir'] = f'{base_tf_log_dir}/epochs={epochs}_lr={str(lr)}_a={0}_b={1}'
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
