import subprocess
import json

runs = [
    { 'ae_latent_dim': 32,  'vecnorm': True  },
    { 'ae_latent_dim': 32,  'vecnorm': False },
    { 'ae_latent_dim': 64,  'vecnorm': True  },
    { 'ae_latent_dim': 128, 'vecnorm': True  },
]

config_path = './exps/FeCAM_cifar100.json'

for run_params in runs:
    with open(config_path, 'r') as file:
        data = json.load(file)
    for param_name, param_value in run_params.items():
        data[param_name] = param_value
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Running FeCAM with params {run_params}')
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    