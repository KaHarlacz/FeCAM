import subprocess
import json

pca_components = [50, 100, 200, 300, 400, 500]

config_path = './exps/FeCAM_cifar100.json'

for pca_comps in pca_components:
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['ae_pca_components'] = pca_comps
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    