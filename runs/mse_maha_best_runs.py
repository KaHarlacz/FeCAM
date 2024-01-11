import subprocess
import json
import datetime

epochs = 1000
runs = [
  { 'a': 1, 'b': 50, 'lr': 0.0001 },
  { 'a': 1, 'b': 100, 'lr': 0.0001 },
  { 'a': 5, 'b': 100, 'lr': 0.00001 },
  { 'a': 1, 'b': 100, 'lr': 0.001 }
]
layers_neurons = [256, 128, 64, 16]
grid_name = 'ae_maha_mse_best'

config_path = './exps/FeCAM_cifar100.json'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_tf_log_dir = f"tf_logs/{grid_name}/" + current_time

for run_params in runs:
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['epochs'] = epochs
    data['lr'] = run_params['lr']
    data['maha_alpha'] = run_params['a']
    data['maha_beta'] = run_params['b']
    data['layers_neurons'] = layers_neurons
    data['tf_dir'] = f'{base_tf_log_dir}/epochs={epochs}_lr={str(run_params["lr"])}_a={run_params["a"]}_b={run_params["b"]}'
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()