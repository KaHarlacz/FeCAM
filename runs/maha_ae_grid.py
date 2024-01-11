import itertools
import subprocess
import json
import datetime

epochs = 100
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.0001]
alpha = [0, 0.1, 1, 10, 50, 100]
beta = [1, 10, 50, 100]
config_path = './exps/FeCAM_cifar100.json'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_tf_log_dir = "tf_logs/ae_rec_cost_maha_grid/" + current_time

for (lr, a, b) in itertools.product(learning_rates, alpha, beta):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['lr'] = lr
    data['maha_alpha'] = a
    data['maha_beta'] = b
    data['tf_dir'] = f'{base_tf_log_dir}/lr={str(lr)}_a={a}_b={b}'
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
