#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available if not use CPU
print(f"Using device: {device}")    


import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
# Args selections
start_time = datetime.now() #set time for logging



parser = argparse.ArgumentParser() # Create an argument parser

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str, #uses the default experiment description
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,  #uses the default run description
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int, #uses the default seed value
                    help='seed value')
parser.add_argument('--training_mode', default='train_linear', type=str, #ændre til forskellige typer af træning
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str, #ændre til forskellige typer af datasæt
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,  #ændre til forskellige typer af logs
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,  #ændre til forskellige typer af devices
                    help='cpu or cuda') # cpu or cuda change to cuda if you have a GPU
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available if not use CPU
print(f"Using device: {device}") 

experiment_description = args.experiment_description  # Experiment description
data_type = args.selected_dataset  # Dataset
method = 'TS-TCC' 
training_mode = args.training_mode 
run_description = args.run_description 

logs_save_dir = args.logs_save_dir # Directory to save logs
os.makedirs(logs_save_dir, exist_ok=True) # Create the directory if it does not exist


if data_type.startswith("EpilepsyCustom_"):
    from config_files.EpilepsyCustom_Configs import Config as Configs
else:
    exec(f'from config_files.{data_type}_Configs import Config as Configs')

configs = Configs() 

# ##### fix random seeds for reproducibility ########
SEED = args.seed # Seed value
torch.manual_seed(SEED) # Set seed for torch
torch.backends.cudnn.deterministic = False 
torch.backends.cudnn.benchmark = False 
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}") # Create the experiment log directory
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log") # Create a log file
logger = _logger(log_file_name) # Create a logger
logger.debug("=" * 45) # Log the experiment details
logger.debug(f'Dataset: {data_type}')  # Log the dataset
logger.debug(f'Method:  {method}')  # Log the method
logger.debug(f'Mode:    {training_mode}')  # Log the training mode
logger.debug("=" * 45)  # Log the experiment details

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)  
temporal_contr_model = TC(configs, device).to(device)

if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models")) # load the self-supervised model
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" and data_type == "Epilepsy":
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.



model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")