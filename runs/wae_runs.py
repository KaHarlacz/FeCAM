import subprocess
import json

ae_type = 'wae'
epochs = 2000
vecnorm = False
wae_c = 0.1
wae_sigma = 10

runs = [
    { 'ae_standarization': False, 'ae_latent_dim': 32 },
    { 'ae_standarization': False, 'ae_latent_dim': 16 },
    { 'ae_standarization': True,  'ae_latent_dim': 32 },
    { 'ae_standarization': True,  'ae_latent_dim': 16 }
]

config_path = './exps/FeCAM_cifar100.json'

for run_params in runs:
    ae_standarization = run_params['ae_standarization']
    ae_latent_dim = run_params['ae_latent_dim']

    with open(config_path, 'r') as file:
        data = json.load(file)
    
    data['ae_standarization'] = ae_standarization
    data['ae_latent_dim'] = ae_latent_dim

    data['ae_type'] = ae_type
    data['epochs'] = epochs
    data['vecnorm'] = vecnorm
    data['wae_C'] = wae_c
    data['wae_sigma'] = wae_sigma

    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Running FeCAM with params C={ae_standarization}, sigma={ae_latent_dim}')
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    